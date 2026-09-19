# 08 — Collections treatment assignment · design sketch

An ideal-world sketch of what this project would look like if the authoring
surface could be anything. Nothing here runs — every body is `pass  # what it
does`. The deliverable is a **shape**.

Read alongside [`FRAMEWORK-DEMANDS.md`](FRAMEWORK-DEMANDS.md), which is the more
useful half: the numbered list of things this project needs from `decider2`,
each traced to the spec section that forced it and marked *satisfied by doc 03*,
*needs extension*, *doc 03 would make this ugly*, or honestly unresolved.

The spec is `../../08-collections-treatment.md`. Section references below
(§5.5, §13.1) are to it unless they say "doc".

---

## 1. Six decisions, and everything else follows

If you read nothing else:

**1. Position in an escalation path is a fourth core-component kind, not a
decision table wearing a hat.** `path/sequence.py`'s `CollectionsPath =
Sequence(...)` folds a small carried state (`PathState`: nine fields, no more)
over an episode's events through a **closed effect algebra** — `advance_to`,
`hold`, `raise_to`, `reset_to`, `clear_counters`, `carry_counters`,
`set_cooling_off`, and nothing else. That closure is what makes "escalation
never decreases outside a reset" a build-time proof (`monotone("intensity_
ceiling", ...)`) rather than a hope tested on a corpus. This is the direct
answer to §13.1: composed from scorecard, table and tree, the hard 70% —
the fold, the resets, the roll carry-over, the monotonicity invariant — would
be hand-written forty times, once per bucket × family path.

**2. The treatment matrix is not doc 03's `Table`.** `matrix/grid.py`'s `Grid`
replaces the four-line, "lowest-confidence part of this document" sketch
wholesale: sparse authoring with most-specific-wins resolution (`71` CSV rows
covering `5 376` cells), coverage **proven, not tested**
(`Grid.validate()` expands the full key space before anything runs), unused
cells **declared**, not inferred (`coverage: traffic | sparse | none`), three
cohort-scoped versions coexisting, and a diff that is the actual reviewable
artefact (`Grid.diff(v213, v214)` → changed cells, prior/new values, accounts
affected, capacity impact per pool).

**3. Suspensions evaluate completely and compile branchless.** `Panel`
(`suspensions/panel.py`), not `Branch` — twenty independent predicates into
one `u4` bitmask at ~14 ns/row, with an evidence ledger emitted only for the
bits that fired. Composition of "which suspensions are in force" is a
**meet-semilattice** (`Scope.meet`), so §9.2's "debt review AND outside hours
AND SMS consent withdrawn" is answerable without caring which predicate the
compiler happened to emit first.

**4. Capacity allocation is a new frame-tier combinator, and non-selection is
materialised because materialising it is cheap.** `Allocate`
(`allocation/constraints.py`) is not composed from `join`/`aggregate`/`filter`
— it is a declared, ordered, four-stage population constraint
(`Floor` → `Reserve` → `rank_fill` → `Pace`) whose stage-that-placed-an-account
**is** its explanation. The 2.3M-account split: 629,000 accounts carry a rank
and (for 286,000 of them) a cut-off; the rest need no rank at all. ~14 MB/day,
36 GB over seven years. §13.6's "how do you explain non-selection without
materialising a rank for 2.3M accounts every day" has the answer built in: you
do materialise it, for the fraction that needs it, and it is cheap.

**5. Overlays are a fourth change class.** Neither a value, an interior, nor
skeleton — free at runtime like a value (`OverlayStack`'s register is arrays),
approved like an interior, audited like neither. The permission boundary is
one file, `overlays/surfaces.py`: a value is overlayable only by being
registered as an `OverlaySurface`, and every statutory parameter in this
project is simply never registered. There is no wildcard, no dotted path, no
authority level that reaches one.

**6. Batch and real-time are the same object.** `pipelines/live_call.py`:
`live = daily.without(population_dependent=True).for_record()`. The one stage
excluded — `CapacityAllocation` — is excluded because it is marked
`@population_dependent`, not because a second pipeline definition says so.
Nobody hand-maintains a live variant that can drift from the batch one,
because there is no second pipeline to maintain.

---

## 2. The layout

```
08-collections-treatment/
  timelines/                 # every event source and every window, declared once
    schemas.py                  # ContactAttempt, PaymentTransaction, Promise, ...
    streams.py                  # Timeline(grain=, event_date=, known_at=, late_by_p99=)
    windows.py                  # 41 windows: rolling(), months(), since(), until()
    calendar.py                 # BusinessDayCalendar Table -> a derived dense index
  state/
    assembly.py                  # AccountState — the ~190-field state_vector contract
    episode.py                   # episode_id and re-entry class, folded, not stored
  matrix/
    dimensions.py                 # the 5 Bands, matrix_cell_id, banding_version
    grid.py                       # Grid — the sparse, diffable, versioned N-D table
    treatment_matrix.v214.csv     # 71 rows authoring 5 376 cells
  path/
    sequence.py                   # CollectionsPath — the Sequence, its checkpoint
    transitions.json              # the interior: 16 rules, an analyst edits this
    intervals.yaml                 # per-channel min-interval / max-attempts / caps
  suspensions/
    panel.py                       # Panel, Scope meet-semilattice, expiry union
    contactability.py              # 115-120, client-grain
    notices.py                     # 112, precondition() on legal handover
    prescription.py                # 114, the asymmetric one
    status_feeds.py                 # 101-111, 113, from the status/legal feed
  cohorts/
    assignment.py                  # stable_hash64 champion/challenger, holdout
  overlays/
    surfaces.py                     # the ONLY registry of what may be overlaid
    guard.py                        # OverlayStack, authoring/activation guards
    register.json                   # the live register: 7 overlays, 1 lapsed, 2 rejected
  allocation/
    pools.py                        # 12 declared Pools, sub-grains, pacing
    ranking.py                      # Choice — two switchable ranking bases
    constraints.py                  # Allocate: Floor/Reserve/rank_fill/Pace/Quota
    capacity.2026-09-19.json        # the one artefact that changes daily
  arrangements/
    authority.yaml                  # L1-L5 authority grid, 6 arrangement types
    distressed.py                   # profile() DISTRESSED, income-linked step
  settlement/
    discount_grid.csv               # bucket x recovery band x authority
    justification.py                # the reproducible expected-recovery comparison
  pipelines/
    daily_batch.py                   # the 04:15 run, 2.3M accounts
    intraday.py                      # the 13:30 re-run, diff + withdrawal
    live_call.py                     # the agent's 400ms path, a slice of daily
```

`evidence/` and `scoring/` are referenced from `pipelines/daily_batch.py`
(`AssignmentRecord`, `SuspensionAttestation`, `CollectionsScorecard`,
`RecoveryCurves`, `ContactBanding`) but are not part of this project. They are
project 09's replay harness and project 00's shared scorecard, consumed the
way project 00 §7 intends — imported, not re-implemented.

### Why this layout, in three sentences

**`timelines/` and `matrix/`/`path/`/`suspensions/` are separated by what they
own**, not by taste: `timelines/` is the only place an event log is read, and
everything after it reads only the ~190-column state vector `state/assembly.py`
assembles from those timelines. That boundary is what makes `pipelines/
live_call.py` possible at all — the live path can patch a state vector
incrementally because the batch path never lets anything downstream reach
past it to a raw event.

**`overlays/` is a sibling of `matrix/`, `path/` and `allocation/`, never a
subdirectory of any of them.** Doc 08 §2 and spec §5.4/§6 are explicit that an
overlay must never be merged into the artefact it modifies; putting the
register inside the matrix directory is the first step toward someone doing
exactly that "to tidy up".

**`arrangements/` and `settlement/` both import from `matrix/grid.py`**
(`ARRANGEMENT_MINIMUMS`, `DISCOUNT_AUTHORITY`) rather than declaring their own
lookup mechanism, because a grid is a grid whether it holds a treatment code or
a discount percentage — the same six properties (sparse authoring, proven
coverage, declared unused cells, coexisting versions, generic-kernel
evaluation, diffable) apply to all four grids in this project.

---

## 3. A daily run, account to instruction

The whole flow is written down once, in order, in `pipelines/daily_batch.py`,
and the file's own header states why the order is load-bearing in one place:

```
matrix -> overlays -> sequence -> SUSPENSION GATE -> allocation
```

Spec §5.4: *"Where a commercial overlay and a suspension would disagree, the
suspension wins by construction rather than by evaluation order, and the
evidence must show that the suspension was evaluated against the overlaid
recommendation, not against the raw one."* Those two sentences pull opposite
ways and the pipeline expression satisfies both at once: the `Gate` is
downstream of the overlay stack (so the evidence shows the *overlaid*
recommendation being refused), and the suspension wins by construction because
`SuspensionPanel` computes a permitted set from the account's own status,
**independently** of the recommendation, and `Gate` only intersects. There is
no ordering in which a commercial overlay could win, because there is no
expression in which the two meet as peers.

Follow one account — 138 days past due, secured asset, R340,000 outstanding,
balance band 5, no suspensions, cohort 41 (challenger arm of the bucket-3
negotiator experiment, assigned before the account rolled into bucket 6).

1. **Assembly** (`Assembly = AssembleState | Episodes | bureau_state_code`).
   180M contact rows, 27.6M payments, 55.2M rolls collapse to ~190 scalars.
   This account: `times_cured_12m = 0`, `episode_id` derived from the roll that
   opened this episode 138 days ago, `re_entry_class = 0`.

2. **Risk** (`fuse(CollectionsScorecard | CollectionsOverlays.on("collections.
   score") | ContactBanding | CollectionsOverlays.on("collections.band") |
   RecoveryCurves)`). Scored even though the account may end up suspended or
   below the capacity cut-off — the estimate is needed for *ranking*
   regardless. `collections_score_unadjusted` and the adjusted score both
   survive. `fuse()` here and nowhere else: five arithmetic-heavy modules with
   no branching (doc 02 §1.1's fusion-positive shape); everything after this
   branches hard and is deliberately left unfused.

3. **Matrix** (`Cohorts | cell_id | band_edge_flag | lookup_treatment |
   CollectionsOverlays.on("matrix.intensity", "matrix.retries", "matrix.
   cooling_off", "matrix.treatment_availability")`). `arrears_bucket_code = 6`,
   `balance_band_code = 5` → the matrix row `6,*,5-7,*,*,10,5,1,0,traffic`:
   pre-legal notice, intensity 5, no retries, zero cooling-off. Overlay 4203
   (bucket-3 intensity dial) doesn't apply — wrong bucket — but overlay 4302
   (allocation weighting for balance bands 6-7) will matter downstream.

4. **Path** (`CollectionsPath | test_intervals | cooling_off_remaining |
   expected_next`). The account has been at path position 4 for 6 days with 2
   attempts against `permitted_retries = 1` — `ESC01` fires: advance to
   position 5, raise the intensity ceiling (already at the matrix's 5, so no
   change), clear the position's counters. `escalation_rule_id = "ESC01"`.

5. **Suspensions** (`Prescription | notice_period_expires_on |
   SuspensionPanel`). All twenty predicates run; none fire.
   `permitted_treatment_mask` is the full set, `suspension_mask = 0`, and the
   panel row records `evaluated=True` regardless — spec §7's "no suspension
   applied" must be distinguishable from "suspensions not evaluated".

6. **Refusal** (`Gate(permitted=..., ceiling=..., proposes=(...))`). Nothing to
   refuse or downgrade here; `gate_outcome_code` records that explicitly.

7. **Settlement** — not proposed today; no right-party contact yet this cycle.

8. **Allocation** (`Ranking | CapacityAllocation`). Balance band 5 doesn't
   qualify for the band-6/7 reserve, so this account competes in the
   value-ranked fill for `notices` (treatment 10's pool). It ranks 4,110th of
   14,500 demand against 12,000 supply — above the cut-off. `allocated = True`,
   `pool_code = notices`, `allocation_stage_code = 3` (rank_fill).

9. **Output.** `treatment_instance_id` minted; `matrix_version=214`,
   `matrix_cell_id`, `cohort_code=41`, `adjustment_set_version=91`,
   `escalation_rule_id="ESC01"`, `expected_next_treatment` computed from
   `CollectionsPath.peek(state, assuming_outcome="non_engagement")` — the
   negotiation-envelope sentence an agent would say if this account rang in
   tomorrow: *"if we don't hear from you, the account moves to legal
   handover on day 151."*

Measured shape from the file's own comments: 6 frame-tier passes / 11 of the 90
minutes in Assembly; the rest is record-tier and cheap by comparison.

---

## 4. An agent on a live call, in 400 ms

`pipelines/live_call.py` is composition, not logic — its own docstring is
explicit that "this file must contain no logic". The two things an agent asks
for are entry points over the **same** pipeline object:

```python
live = daily.without(population_dependent=True).for_record()

assess_arrangement = live.entry(
    name="assess_arrangement", adds=DistressedAffordability,
    inputs=("proposed_instalment_cents", "proposed_duration_months",
            "arrangement_type_code", "agent_authority_level"),
    returns=("decision_code", "required_authority_level", "minimum_instalment_cents",
             "max_duration_months", "affordability_verdict_code", ...))
```

`without(population_dependent=True)` drops exactly `CapacityAllocation` — the
one stage marked `@population_dependent` — and nothing else, because that
marker is declared rather than inferred (FRAMEWORK-DEMANDS #14). An agent
asking "can I accept this arrangement" is not asking whether the account would
win an allocation, so dropping it is correct, and it is dropped **by a rule**,
not by a second hand-written composition somebody has to remember to update
when `daily` changes.

At 10:14:33 the batch state assembled at 04:20 is patched, not rebuilt:

```python
state = AccountState.patch(batch_state,
                           since=batch_state.knowledge_cutoff,
                           now=request_knowledge_cutoff)
```

This works only because `AccountState` is declared `@state_vector(realtime=True)`
(`state/assembly.py`) — every field has a declared incremental form, and the
**build fails**, naming the field, if someone adds one that doesn't. Three
payments and one contact have landed since the morning; the patch re-runs only
the windows whose timelines moved.

The measured budget from the file's own comments:

| stage | cost |
|---|---|
| fetch batch state vector (1 row, key lookup) | 8 ms |
| fetch deltas since watermark (4 timelines) | 21 ms |
| incremental window patch | 2 ms |
| kernel: risk + matrix + path + suspensions + gate | 0.3 ms |
| arrangement / settlement assessment | 0.2 ms |
| evidence assembly and response | 6 ms |
| **total** | **~38 ms p50, 140 ms p99** |

`negotiation_envelope(state, rt)` is what actually goes on the agent's screen
before anything is proposed: minimum acceptable instalment, maximum duration,
maximum discount at the agent's own authority, promise limits, what needs
referral — computed by the **same grids** the acceptance path reads, in the
same call, so the envelope and the decision cannot disagree. It also carries
`expected_next_treatment` from `CollectionsPath.peek()` — the same object that
will make tomorrow's batch decision — because the sentence the agent says out
loud ("if we don't hear from you by Thursday...") is a promise the Bank then
has to honour or explain, and it has to come from the thing that will actually
decide Thursday's treatment.

The nightly 5,000-account reconciliation (NFR "Batch/real-time agreement")
tests something narrower than "two implementations agree": it tests that the
incremental patch produced the same state vector the batch assembled. That is
a data question, not a logic question, because there is only one `Decision`-
shaped kernel and it cannot disagree with itself.

---

## 5. The hard parts, and how each is expressed

### 5.1 Time, expressed so it replays

Every window in the project is declared as data in `timelines/windows.py` —
41 of them, from `paid_30d` to `automated_only_streak_days` — over `Timeline`
objects declared in `timelines/streams.py`. Two fields do all the work: a
window filters on

```
event_date <= decision_date  AND  known_at <= knowledge_cutoff
```

Setting `knowledge_cutoff` to the run's watermark gives the as-known-on-the-day
derivation audit needs; setting it to `Timestamp.MAX` gives the as-at-now
derivation analysis wants (spec §5.5, disagreeing on 2.4% of account-days) —
**one implementation, two values of one field**, not two code paths.
Business-day arithmetic — every notice period, every grace end, every "within
N business days" test — resolves through `timelines/calendar.py`'s dense
`ORD`/`INV` int32 arrays, so `bdays_between` and `add_bdays` are one subtract
and one index, O(1) inside a numba kernel with no data-dependent loop.

### 5.2 Treatment as a position in a sequence, not a fresh decision

`path/sequence.py`'s `CollectionsPath` is a `Sequence`: a fold over an
episode's events, carrying `PathState` (nine fields — `path_position`,
`intensity_ceiling`, `intensity_floor`, `attempts_at_position`,
`channel_attempts`, `cooling_off_until`, `last_reset_code`, `last_reset_on`).
The interior — `path/transitions.json`, 16 rules an analyst edits — can only
call seven declared effects (`advance_to`, `hold`, `raise_to`, `reset_to`,
`clear_counters`, `carry_counters`, `set_cooling_off`); there is no
`set_position = <expression>`. That closure is what lets
`monotone("intensity_ceiling", direction="non_decreasing",
except_after=RESET_EVENTS)` be a **build-time proof over the closed effect
algebra** rather than a property somebody remembers to test — a rule that
could lower intensity outside a reset fails the build naming the rule id.

Folding 90 days of events for 2.3M accounts every morning would blow the
90-minute window, so the fold is checkpointed: `checkpoint_key = (episode_id,
transitions_version, intervals_version)`, and a checkpoint is resumable only
if every contributing timeline's watermark hasn't moved backwards past it.
Agency outcomes land T+2, so a checkpoint younger than `late_by_p99` is
`provisional`, and any event landing behind it forces a replay of the episode
from its open date — measured 96.4% resume, 3.6% replay, same code path either
way. Because the checkpoint carries the rule *versions* that produced it, a
replay in 2033 with the 2026 `transitions.json` reproduces the 2026 position
without ever consulting the checkpoint at all — which is the actual answer to
spec §13.2's "is temporal state stored or derived": **both**, and the cost of
the other is bounded.

The episode itself is never a stored column, for the same reason:
`episode_id = stable_hash64(account_id, episode_opened_on)`
(`state/episode.py`) is content-derived, so it is the same integer whoever
computes it — including the live-call path, in about 40 microseconds, with no
database lookup.

### 5.3 A 5,376-cell matrix an analyst actually owns

`treatment_matrix.v214.csv` authors all 5,376 cells in 71 rows using `*`
wildcards and `a-b` ranges, resolved most-specific-wins under a **declared**
priority order (`arrears_bucket_code` first, then `product_family_code`, ...).
Two rows of equal specificity that overlap are a build error naming both, so
"which row won" is never a question and the file's row order is never
load-bearing — it can be sorted, merged, diffed freely. `Grid.validate()`
proves total coverage before anything runs, and `coverage: traffic | sparse |
none` on every row makes the ~41% sparse and ~12% empty cells a **declaration**
the monthly exercise report checks against reality, in both directions — a
cell declared `none` that took traffic is exactly as reportable as one
declared `traffic` that didn't (acceptance criterion 11). `banding_version` is
threaded through `matrix_cell_id` (`matrix/dimensions.py`) so that change
scenario 1 — an interior change that splits bucket 8 and takes the matrix from
5,376 to 6,048 cells — leaves eighteen months of history under the old banding
interpretable rather than silently mixed with the new one.

### 5.4 More work than capacity, ranked and rationed fairly

`allocation/constraints.py`'s `CapacityAllocation = Allocate(...)` runs four
ordered stages over nine constrained pools (`allocation/pools.py`): mandatory
`Floor`s that bind before value (the 21/30-day untouched-days floor, the
bucket-3 coverage floor, the channel-starvation floor, the pre-prescription
legal pull), a `Reserve` for high-balance bands, `"rank_fill"` by a
**switchable** ranking basis (`Choice(param="ranking_basis", options={1:
marginal_value_rank_key, 2: policy_priority_rank_key})` — spec §5.9's
requirement that the basis be changeable "without a release"), and `Pace` for
the monthly-paced pools. Cohort neutrality is structural, not assumed: each
pool's capacity is split across cohorts in proportion to demand share, and the
residual rounding imbalance is a **reported** number
(`neutrality_report="cohort_rationing_imbalance"`), which is what acceptance
criterion 9 means by "demonstrably uncorrelated" rather than "assumed".

The stage that placed an account **is** its explanation
(`allocation_stage_code`), which is what turns non-selection code 250 from a
population-level fact into a per-account one: *"you were 91,412th of 186,000 in
the early agent pool, and we stopped at 27,000"* is a stored fact, not a
reconstruction, because on an observed day only 629,000 of 2.3M accounts ever
reach stage 3 and need a rank at all (`allocation/constraints.py`'s own
accounting: ~14 MB/day, 36 GB over seven years).

### 5.5 Suspensions that override everything, individually attributable

`suspensions/panel.py`'s `Panel` evaluates all twenty suspensions
unconditionally — no short-circuit — because the predicates are declared
mutually independent (a build error if one reads another's output), which is
what licenses branchless codegen: ~14 ns/row for twenty predicates, because the
expensive part was never the comparison, it was assembling evidence, and
evidence is emitted only for the bits that fire. Composition is a
**meet-semilattice** (`Scope.blocks_all`, `blocks_channels`, `blocks_treatments`,
`downgrades_to`, and `meet`, which is commutative, associative and idempotent)
so four suspensions in force compose to one answer regardless of which
predicate the compiler happened to emit first — the direct answer to §9.2's
audit question about a client under debt review, outside contact hours, with
SMS consent withdrawn.

Expiry is a required, three-way tagged union — `computed(...)`, `event(...)`,
`boundary(...)` — so "suspended indefinitely" is not a representable state
(spec §5.2). Prescription (114, `suspensions/prescription.py`) is the one that
runs the wrong way: it `enables=(PRESCRIPTION_NOTIFICATION,)` rather than only
blocking, which is the declared hook change scenario 10 (a court-ordered
proactive notification) needs without redesigning the suspension model. And
`precondition()` (`suspensions/notices.py`) makes "no legal handover without
served-notice evidence" structural rather than a rule someone could forget to
write: `on_failure="record_attempt"` means the framework itself emits the
attempted-handover evidence row when the precondition fails, which is what
acceptance criterion 12 and §9.4's "not a sample" actually require.

### 5.6 Overlays that tilt commerce, never statute

`overlays/surfaces.py` is deliberately the only file that answers "what can be
overlaid": a value becomes overlayable **only** by being declared an
`OverlaySurface`, and every statutory parameter in the project — suspension
rules, notice periods, contact hours, frequency caps, prescription, the
discount authority grid — is simply never registered. `register.json`'s
`rejected_attempts` block is not decoration; it is spec §5.4's "the attempted
definition is itself recorded" made real:

```json
{ "attempted_target": "suspension.117.cap_per_7d", "attempted_magnitude": {"set": 14},
  "rejected_because": "not_an_overlay_surface", "owner_of_target": "regulatory_compliance" }
```

Where a surface exists but only one direction is safe, the combiner enforces
it structurally: `TIGHTEN_ONLY(direction="increase")` on cooling-off means an
overlay proposing 2 days over a base of 7 resolves to 7 and is recorded as a
**no-effect application** — not silently discarded, not silently loosened.
`on_review_date_lapsed="expire_and_alert"` means an overlay past its review
date **stops applying** rather than merely triggering a report nobody reads;
`register.json`'s overlay 4204 shows the mechanism three days before it fires.
The weakest of the three enforcement mechanisms is honestly named in
`overlays/guard.py`: `allocation.weight` overlays are validated by *running
the allocator* over a reference population at authoring time (a real check,
not a proof), which is FRAMEWORK-DEMANDS #16 and the ugliest coupling in the
sketch.

### 5.7 One affordability capability, two modes

`arrangements/distressed.py` answers acceptance criterion 7 with a `profile()`
— `DISTRESSED = profile(id="core.affordability:distressed_collections",
version=4, values={...})` — a named, versioned, effective-dated, **registered**
bundle Collections references by id rather than by forty loose values that
happen to have come from somewhere. Two of the three differences between
granting and distressed are numbers (a `profile` handles those); the third —
whether the statutory expense norm is a disqualifier or a recording floor — is
structural, so it is a `Branch` (`ExpenseNormTreatment`) with **both arms in
the graph and in lineage** whichever one runs, never a scalar `mode` flag that
would hide one arm from the other project's reviewer. `affordability_mode_code`
is a required output precisely because criterion 7 says the mode must be
*named* in the evidence, not implied by which numbers came out.

### 5.8 Champion/challenger without becoming an overlay

`cohorts/assignment.py`'s `assign_cohort` is `stable_hash64(account_id, salt)`
— never Python's `hash()`, which is `PYTHONHASHSEED`-salted per process and
would silently re-randomise the split between the 05:00 batch and the 13:30
re-run while looking perfectly deterministic in any single test run. The
file's own comment states the three structural separations from overlays that
keep the two mechanisms from being conflated (spec §5.12's central worry):
different artefacts (an `Experiment` has `arms`/`measures` and no `magnitude`;
an overlay has `magnitude`/`stack_position` and no `arms`), different reach (a
challenger selects a matrix *version*; an overlay modifies a *value* the
matrix produced), and different neutrality obligations (`cohort_scoped: false`
on an overlay triggers a demonstrable-equality check; `cohort_scoped: true`
marks every experiment it touches `arms_not_comparable`). The standing holdout
carries a mandatory expiry like everything else in the register — no exception
for measurement traffic.

---

## 6. What a Collections Strategy analyst sees and edits

| Analyst owns | File | Cannot touch |
|---|---|---|
| Treatment matrix, monthly + mid-month patches | `matrix/treatment_matrix.v214.csv` | `treatment_code` via overlay — only intensity, retries, cooling-off, availability |
| Matrix intensity dials, weekly (Collections Credit Forum) | `overlays/register.json` | Anything in `overlays/surfaces.py`'s absent list — no name reaches a suspension |
| Ranking basis | `allocation/ranking.py`'s `ranking_basis` param | The tie-break (`rank_tie_break` is code, stable-hash, never touched) |
| Contact interval rules, quarterly, Compliance co-signs | `path/intervals.yaml` | The absolute per-client caps at the bottom — Compliance-owned, `overlayable: false` |
| Path transitions, when policy on resets/escalation changes | `path/transitions.json` | The seven-effect vocabulary itself — that's code |
| Champion/challenger experiments | `cohorts/assignment.py`'s `REGISTRY` | `assign_cohort`'s signature — `account_id` and nothing else, deliberately |
| Band edges (semi-annual/quarterly) | params behind `matrix/dimensions.py`'s `Band(...)` declarations | The five dimensions themselves — a sixth dimension is a skeleton change |

What she edits is a value or an interior; what she can never reach is
registered, closed, or simply not in her file. `arrangements/authority.yaml`
and `settlement/discount_grid.csv` are Credit Risk Policy's, not hers — the
directory boundary is the ownership boundary, and `arrangements/distressed.py`
importing `matrix/grid.py`'s `ARRANGEMENT_MINIMUMS` rather than declaring a
parallel lookup is what keeps "who signs this" answerable from the file tree
alone.

The one artefact she actually signs is `Grid.diff(v213, v214)` — 47 changed
cells (`treatment_matrix.v214.csv`'s own header comment), each with prior and
new values, accounts affected last month, and expected demand impact per
capacity pool. Everything else in the matrix directory is machinery to make
that diff trustworthy.

---

## 7. "Why was I contacted" — and why weren't you

§9.1's ombud question — *"the full contact history and the rule that
authorised each contact"* — is answered per contact from three columns that
already exist on every `TreatmentInstance`: `matrix_cell_id` +
`banding_version` (which cell, under which key domain), `escalation_rule_id`
(which line of `path/transitions.json` decided today, e.g. `"ESC01"`), and
`applied_overlay_ids` in stack order with the effect of each. Nothing is
reconstructed after the fact; every field was written the day the contact was
decided.

§9.2's *"you contacted her during debt review"* is answered from the
`SuspensionPanel`'s evidence ledger: whether debt review was known, when it
arrived (`known_at` against the run's `knowledge_cutoff`), what stage, and —
if contact was permitted — the exact exception (statutory notice, e.g.) that
allowed it. "The status arrived after the run" is provable from the feed
watermark carried on the window, not asserted.

§9.5's *"why 40 days of silence"* is answered by reading the non-selection
record for each of those 40 days. `allocation/constraints.py`'s
`NON_SELECTION_CODES` table is the closed vocabulary: 200 (matrix said do
nothing), 210 (suspended, naming every code that applied), 220/230/240
(cooling-off, interval, promise, each with the specific detail spec §5.9
requires), 250 (below the capacity cut-off — pool, rank, ranked count,
cut-off rank, ranking basis), 260 (a sibling account of the same client was
actioned instead — the per-client harassment defence), 270 (no permitted
channel). Forty days of code 250 is forty rows an ops manager can read as "the
early-agent pool never had headroom for this account", which is a completely
different remediation from forty days of code 220.

---

## 8. What changes when a value moves, versus when structure moves

| change | class | artefact | cost | who / how long |
|---|---|---|---|---|
| A matrix cell's intensity or cooling-off | value (overlay) | `overlays/register.json` | bundle swap, microseconds | Collections Credit Forum, weekly |
| A band edge (e.g. the 3/4 arrears cutoff) | value | params behind `Band(...)` | bundle swap | Collections Strategy, semi-annual |
| Daily agent hours / capacity | value | `allocation/capacity.2026-09-19.json` | bundle swap, every morning | Workforce Management feed |
| Ranking basis (value → policy) | value | `ranking_basis` param | bundle swap, no release | Collections Strategy |
| A cell's `treatment_code` (a full matrix version) | interior | `treatment_matrix.v214.csv`, new version | one background compile + staged swap | Collections Strategy, monthly + reviewed diff |
| A path-transition rule's `when`/`then` | interior | `path/transitions.json`, new version | one background compile + staged swap | Collections Strategy, co-signed where statutory codes are touched |
| A suspension's threshold (e.g. `cap_per_7d`) | interior, Compliance-owned | Compliance's own params document | bundle swap, but on Compliance's release path | Regulatory Compliance, on regulatory change (change scenario 3) |
| A ninth arrears bucket (change scenario 1) | interior + skeleton | `Band` edges/labels, 672 new cells, `banding_version` bump | validator blocks release until every cell populated | Collections Strategy + engineer to bump `banding_version` |
| A new treatment (change scenario 2) | skeleton | new pool, new `OverlaySurface` scope value, new matrix column value | rebuild and redeploy | Engineer |
| An income-linked arrangement type (change scenario 7) | skeleton, narrowly | one registered step referenced by id from a nullable grid column | rebuild and redeploy for the step; the grid row itself is a value | Engineer once, analyst thereafter |
| Intraday re-run becomes hourly (change scenario 13) | scheduling, not skeleton | nine invocations of `rerun(daily, ...)` with nine cutoffs | none — it is the same object | Collections Operations |

The line that matters is between rows 8 and 9: adding a bucket is expensive
because the *matrix's own key space* grows (a real structural event, correctly
priced at a validator gate and a version bump), while adding a treatment is
expensive because it needs a new pool and a new overlay-surface value — both
genuinely new capacity and governance concepts, not accidental cost. Nothing
in this project makes a common change costly by construction; the costly rows
are the ones the business actually calls "a big deal".

---

## 9. What this sketch does not resolve

Three things, honestly, with the full argument in `FRAMEWORK-DEMANDS.md`.

The `allocation.weight` overlay validator has to **run the allocator** at
authoring time to prove a proposed dial stays feasible against the fairness
floors (#16) — a real check, but a check, and the one place this design's
correctness claim is weaker than the rest of it. The spec's own vocabulary
(§4.1) declares money fields like `arrears_amount` as `float64` while doc 03
§1.2 is unambiguous that money is `int64` cents; the sketch follows the
framework rule and breaks the spec's table in its most visible place (#21).
And the income-linked arrangement type (change scenario 7) is the one place
the grid abstraction genuinely strains — a registered step referenced by id
from a nullable grid column works, but it is a second mechanism sitting next
to the grid's normal value columns, not a natural extension of them (#19).

---

## 10. One inconsistency fixed

`pipelines/daily_batch.py` imports `AssignmentRecord` and
`SuspensionAttestation` from `..evidence.record` — a project-09 module, per
this project's own convention of consuming shared/other-project capabilities
by import (the same line imports `..scoring` two lines below, annotated
`# core.scorecard`). The `evidence.record` import carried no such annotation,
which is inconsistent with the file's own convention and would read as a
missing local file to anyone who didn't already know project 09 owns it. Fixed
by adding `# project 09` to that one line; no other change was made.
