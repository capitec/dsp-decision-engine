# Framework demands — 08 collections treatment

Numbered, traced, marked. **SAT** = satisfied by doc 03. **EXT** = doc 03 lacks
it but nothing in doc 03 fights it. **UGLY** = doc 03 *has* a way and the way is
bad enough that the sketch did something else — these are the ones worth
arguing about. **AWK** = genuine awkwardness the sketch could not resolve and is
not hiding.

---

## Time, windows and replay

**1. `Timeline` — a declared event source with `event_date`, `known_at` and
`late_by_p99`, so "now" is unreachable.** *§5.1 ("resolved against
decision_date and not the wall clock"), §13.3. EXT.*
Doc 03 has `params`/`shared`/leaf inputs and no declared notion that a whole
project's temporal quantities must resolve against one shared instant. A window
here filters on `event_date <= decision_date AND known_at <= knowledge_cutoff`
(`timelines/streams.py`), and `knowledge_cutoff` arrives as a `shared` field
precisely so a compiled step has nothing ambient to reach for. The gap this
closes is real: nothing in doc 03 stops a step author from calling
`datetime.now()`, and a lint for that doesn't exist yet either. `Timeline`
gives the lint something to check against — a step that reads a raw event
table instead of a declared window is visible by inspection.

**2. Time windows as declared data, one frame-tier pass per timeline.**
*§13.10 ("30/60/90-day, 12/24-month, rolling-7-day... one concept, not forty
hand-written filters"), §5.1. EXT.*
`timelines/windows.py` declares 41 of them — `rolling(days=90)`,
`months(24)`, `since("episode_opened_on")`, `until(days=21)` — and the
framework groups every window over the same timeline into one scan. Doc 02 §1
defers `window` from the frame tier on the theory that most window use is
aggregation-plus-join; this project tests that and it holds, but only because
`Window` exists as a *declared combinator* over a `Timeline`, which nothing
in doc 03 or doc 02 §1 currently specifies.

**3. As-known-on-the-day and as-at-now as two values of one field, not two
implementations.** *§5.5 ("disagree on 2.4% of account-days"), §13.3. EXT.*
Setting `knowledge_cutoff` to the run's watermark gives the audit derivation;
`Timestamp.MAX` gives the analysis derivation. Doc 03 has no concept of
"knowledge as of" versus "state as of" as distinct dials on one invocation —
adding it is what turns a genuinely hard requirement into one field's value.

---

## One state per account

**4. `@state_vector` — a declared, versioned, incrementally-annotated record
type spanning many downstream modules.** *§5.1's ~190-field assembled state,
§13.12, NFR "batch/real-time agreement". EXT — and the deviation `state/
assembly.py` names directly.*
Doc 03 has leaf inputs, wired by name, inferred per module — right for a
15-input affordability module, wrong for a 190-field state vector read by nine
downstream modules, because every one of those modules' declared interfaces
would then list forty leaf inputs and a reviewer could no longer see the
boundary. `AccountState` is a named type with a per-field `incremental=`
clause, and the framework **refuses to build** a `realtime=True` pipeline
whose state vector contains a field with no incremental form — a compile-time
guarantee doc 03's inferred-interface model has no way to express, because it
has no boundary object to attach the guarantee to.

**5. Explicit cross-grain broadcast, checked at build time.** *§5.5's 340,000-
client harassment bug. EXT.*
`contacts_7d_client.broadcast(from_grain="client_id")` is a declaration a
reviewer reads in `state/assembly.py`; reading a client-grain window into an
account-grain step without it is a build error. Doc 03's wiring-by-name model
(§2) has no notion of grain, so nothing there would catch three sibling
accounts each silently reading "their own" copy of a value that is shared —
the single sharpest bug this project can produce.

**6. Grain declared per `Timeline` instance, not inferred from a join.**
*§5.2, §5.5 (frequency caps are per-client; sequence counters are per-account,
same event table). EXT.*
`CONTACTS_BY_CLIENT` and `CONTACTS_BY_ACCOUNT` are two declarations over the
same schema. Doc 02's frame tier has `join`/`aggregate`/`filter` but no
concept that the *same physical rows* need two independently-typed views at
two different keys, checked against each other at composition.

---

## Business-day arithmetic is a table, not a callback

**7. A `Table` with a derived index, built from a source artefact.**
*§5.1 ("business days are calendar-dependent and the calendar is a
parameter"), §5.2, §5.8. EXT — the deviation `timelines/calendar.py` names
directly.*
Doc 03 §4's `Table` sketch is `tables.term.max_loan[term]`: a keyed lookup of
a *stored* value, and says nothing about a table that is itself the input to a
derivation. `BusinessDayCalendar` is Compliance-owned gazetted data;
`build_bday_index` turns it into a dense `(ORD, INV)` pair so `bdays_between`
and `add_bdays` are one subtract and one index — O(1) inside a numba kernel
with no data-dependent loop. `numpy.busday_offset` is not compilable in a
kernel and carries no effective-dated holiday calendar, so without this the
single most-used primitive in the project (every notice period, every grace
end) becomes a Python callback that breaks the 400 ms budget the moment an
agent calls.

---

## Sequence position over time

**8. `Sequence` — a fourth core component kind, with a closed effect algebra,
a checkpoint, and a statically-provable monotonicity invariant.**
*§13.1 (is this composed from scorecard/table/tree, or a fourth kind?), §5.5,
§13.2. This is the central question the project exists to answer. EXT.*
It is a fourth kind. Composing it from the three doc 03 already has means a
decision table for interval rules, a tree for the escalation decision, and
hand-written code for the fold, the resets, the roll carry-over and the
monotonicity invariant — the hard 70%, written once per bucket × family path,
by hand. `path/sequence.py` has four load-bearing parts doc 03 has no
equivalent for. Its **unit of evaluation is an episode**, not a record — the
state at `decision_date` is a fold over every event since the episode opened.
Its interior (`path/transitions.json`) may call only seven declared effects —
`advance_to`, `hold`, `raise_to`, `reset_to`, `clear_counters`,
`carry_counters`, `set_cooling_off` — a **closed** vocabulary, which is what
makes `monotone("intensity_ceiling", except_after=RESET_EVENTS)` a build-time
proof over the algebra rather than a corpus-tested property: `reset_to` is
legal only inside a rule whose guard names a reset event, checked
syntactically. It needs a **checkpoint** — folding 90 days of events for
2.3M accounts daily does not fit 90 minutes — keyed on `(episode_id,
transitions_version, intervals_version)` with watermarks from every
contributing timeline, so a checkpoint younger than `late_by_p99` is
`provisional` and a late-landing event forces a replay, same code path either
way (measured 96.4% resume / 3.6% replay). And `CollectionsPath.peek(state,
assuming_outcome=...)` answers "what happens next" from the **same** object
that will make the real decision, so `expected_next_treatment` (spec §5.10,
"something the Bank must honour") cannot drift against a parallel
hand-written rule set. None of doc 03's `Branch`/`Loop` machinery gives any
of these four for free; each had to be designed for this kind specifically.

---

## Suspensions: complete evaluation, individual attribution

**9. `Panel` — a reduction over N independent predicates into one bitmask,
with attribution, at short-circuit performance.** *§13.7 ("short-circuiting
is natural and wrong — what gives complete evaluation at short-circuit
performance?"), §5.2, §9.2. UGLY — the deviation `suspensions/panel.py` names
directly.*
Doc 03's only conditional is `Branch`, an exclusive n-way choice: one arm
runs, `branch_path` records which. Twenty suspensions as nested branches gives
2^20 paths and one integer that names a path rather than a set — unreadable,
and it defeats §9.2's requirement that four suspensions in force each be
individually named. `Panel` declares the predicates mutually independent (a
build error if one reads another's output), which is what licenses branchless
codegen — twenty comparisons, no dependency chain to stall on, ~14 ns/row —
and makes "evaluated, none applied" a recorded fact rather than an inference
from an empty list, which is what acceptance criterion 2 requires.

**10. Suspension composition as a meet-semilattice, so order cannot matter.**
*§9.2 ("does not depend on which predicate the compiler happened to emit
first"), §5.2. EXT.*
`Scope.meet` — `permitted := a.permitted & b.permitted; ceiling := min(a, b)`
— is commutative, associative and idempotent by construction. Doc 03 has
nothing that composes independent effects this way; a hand-rolled
"apply each suspension in sequence" loop would make the answer depend on
iteration order the moment two suspensions both restrict the same channel
differently, which is exactly the failure mode a regulator's audit question
would surface.

**11. Suspension expiry as a required three-way tagged union.** *§5.2
("'suspended indefinitely' is not an acceptable state"), §13.8. EXT.*
`computed(...)` (date arithmetic), `event(...)` (named event + review cadence
+ feed), `boundary(...)` (edge of a recurring window) — a required argument of
`suspension()` with no fourth constructor and no default. Doc 03 has no
notion of an expiry contract at all; without one, "permanent" is exactly as
representable as "expires next Tuesday", which is the state the spec forbids.

**12. `precondition()` — a treatment-level gate that records its own failed
attempt.** *Acceptance criterion 12, §9.4 ("not a sample"). EXT.*
`legal_handover_requires_served_notice` with `on_failure="record_attempt"`
means the framework itself emits the attempted-handover evidence row when the
precondition fails — not a predicate returning `False` that a hand-written
caller might or might not log. Doc 03 has no concept of a postcondition or
precondition attached to a specific downstream action; without one, "no legal
handover without notice evidence" is a rule someone can forget to enforce,
which is precisely wrong for a thing §9.4 says the regulator tests
exhaustively rather than by sample.

---

## Determinism and capacity allocation

**13. `stable_hash64` as a framework primitive, never Python's `hash()`.**
*Acceptance criterion 5 ("re-run of the same decision_date produces identical
assignments"), §5.9, §5.12. EXT — and the single most plausible way to fail
determinism silently.*
`hash()` is `PYTHONHASHSEED`-salted per process: a tie-break built on it is
self-consistent within one test run, then silently re-randomises between the
05:00 batch and the 13:30 re-run, while looking perfectly deterministic to
whoever wrote the test. `allocation/ranking.py`'s tie-break and `cohorts/
assignment.py`'s cohort assignment both need a fixed, documented,
version-pinned hash — a framework primitive, not a convention every author
has to independently avoid getting wrong.

**14. `@population_dependent` — a declared lineage marker for a value that
depends on the whole population, not just this record.** *§13.5 ("its outcome
must be recorded as a per-account explanation"). EXT — `allocation/
constraints.py` names this directly.*
An account's `allocated` flag depends on 185,999 other accounts. Doc 04 §3's
static lineage is a record-tier property — "which inputs and steps can affect
output z" — and has no vocabulary for "affects, but only in aggregate with
everyone else". `@population_dependent(over="pool_code", key="rank_key")`
makes that dependency a declared, greppable fact in the same family as
`@breaks_lineage`, and it is the mechanism `pipelines/live_call.py`'s
`daily.without(population_dependent=True)` reads to correctly exclude
`CapacityAllocation` from the real-time path — a structural guarantee, not a
second pipeline definition someone has to keep in step.

**15. `Allocate` — a new frame-tier combinator with declared, ordered stages
and materialised non-selection.** *§13.5 ("is this the same shape as project
07's portfolio budget, or different?"), §5.9, §13.6. UGLY.*
Doc 03's frame tier ships `join`/`aggregate`/`filter`. None of those is a
top-K-subject-to-constraints, and composing one from them loses the thing
that matters: the per-account explanation. It is the same *shape* as a
portfolio budget and a harder instance — nine constrained pools, fairness
floors that bind before value, monthly pacing, a per-client quota, and a
cohort-neutrality obligation. `Allocate`'s stages (`Floor` → `Reserve` →
`"rank_fill"` → `Pace`) make the stage that placed an account its own
explanation, and §13.6's "how do you explain non-selection without
materialising a rank for 2.3M accounts" turns out to have a cheap answer once
the combinator exists: only the 629,000 accounts reaching stage 3 need a rank
at all (~14 MB/day, 36 GB over seven years) — the instinct to avoid
materialising it is the instinct that makes the ombud question a
data-science project.

**16. A feasibility-simulation overlay validator, which needs the allocator
at authoring time.** *§5.9 ("an overlay that would push any account past the
untouched-days limit is rejected at authoring, naming the constraint"). EXT,
and the ugliest coupling in the sketch.*
`ALLOCATION_WEIGHT`'s `validate=("feasible_against_fairness_floors",)`
(`overlays/surfaces.py`) means an `OverlaySurface`'s authoring-time check is
not a pure function of the overlay document — it has to *run* `Allocate` over
a reference population to check a proposed reserved-share multiplier against
the fairness floors. Doc 03 has no concept of a validator that depends on
another module's compiled behaviour; this is real validation and it can be
wrong if the reference population is unrepresentative, worth stating plainly
rather than presented as a proof. It is also the one place this sketch's
architecture creates a build-time dependency from `overlays/` on
`allocation/`, which the layout otherwise keeps one-directional.

---

## Arrangements: one capability, two modes

**17. `profile()` — a named, versioned, effective-dated, registered bundle of
parameter values, owned by the capability rather than the consumer.**
*Acceptance criterion 7 ("the mode as a parameter... named in the evidence").
EXT — `arrangements/distressed.py` names this directly.*
Doc 08 §6.2's params documents are anonymous dicts with an opaque `origin`
token the framework never parses. That is right for a project's own params,
wrong for a value forty other things might reference: Collections does not
write `{"min_residual": 35000, ...}`, it writes `{"profile":
"core.affordability:distressed_collections@v4"}`, and the values come from
Credit Risk Policy's own document. A profile needs an *identity* that resolves
through the same union as a module id — checked, versioned, effective-dated —
so it appears in the audit record as a reference, not as forty loose values
that happen to agree.

**18. A capability's mode split across a numeric params bundle and a
structural `Branch`, in the same declaration.** *§5.6 ("three things differ
and only one is a number"), acceptance criterion 7. EXT.*
`arrangements/distressed.py` states this precisely: thresholds are a `profile`
(numbers), but whether the statutory expense norm is a disqualifier or a
recording floor is a *branch*, because the two arms have different
consequences over the same inputs. A single scalar `mode` param would hide
that difference from `lineage()`; the `Branch` keeps both arms visible
whichever one runs. Doc 03 §8.2 already gives `Branch` — what's missing is the
guidance that a "mode" is not automatically a parameter, and that the
signature of a genuinely structural difference is "same inputs, different
consequence", not "different number".

**19. A registered step, referenced by id from a nullable grid column, as the
escape hatch for a value a grid's fixed schema cannot express.** *Change
scenario 7 (an income-linked variable instalment that "does not fit the
minimum-percentage grid"). EXT, and not free.*
`ARRANGEMENT_MINIMUMS`'s value schema has no column for a formula, and adding
one would be an expression language in config, which doc 08 §3.2 forbids for
good reason. The answer is a seventh arrangement type whose
`min_pct_of_instalment` is nullable, with a registered step
(`collections:income_linked_minimum`) referenced by id doing the actual
calculation. It works, and it is a second mechanism sitting beside the grid's
normal value columns rather than a natural extension of them — a grid answers
"what value applies here" and this answers "what value applies here, except
sometimes it's actually a function", which a reviewer has to learn as a
special case.

---

## Bands, and a vocabulary contradiction

**20. `Band` — a named, ordered, parameterised edge set, namespaced by a
`banding_version` that survives an interior change to the edges themselves.**
*Change scenario 1 (a ninth bucket, 5,376 → 6,048 cells), §4.3 ("bands are
parameters, not constants, and their edges move"). EXT.*
Moving an edge is a value change (doc 08 §2 class 1) — free, and the impact
report states how many accounts moved cell. Adding an edge is an interior
change: the grid gains cells, a validator names every unpopulated one, and
`matrix_cell_id` is renumbered — which is exactly why cell ids must carry
`banding_version`. Doc 03 has parameters and it has tables; it has no
concept of a *derived key space* whose cardinality itself is a tunable, nor of
what "the same" cell id means across two different key spaces. Without a
namespaced version, cell 3,417 under the old banding and cell 3,417 under the
new one would silently mix in eighteen months of outcome attribution — the
exact hazard change scenario 1 is written to test.

**21. Money is a scaled int64 of cents everywhere — and the spec's own
vocabulary table says `float64`.** *§4.1 declares `arrears_amount: float64`;
doc 03 §1.2 is unambiguous that money is never float. AWK.*
The sketch follows doc 03 and breaks the spec's vocabulary table in its most
visible place. Two things make that the right call rather than a shortcut:
R2 discount rounding must be exact, not approximate, and several accumulators
in this project (the discounted-net-recovery comparison, the pacing carry for
monthly-quota pools) are running sums where an int64 cents accumulator is
exactly the failure doc 03 §1.2 measured wrapping at 2,667 rows. This is
worth fixing in the spec's own §4.1 table rather than papering over here —
every other Bank-fictional spec that reuses this vocabulary (per project 00
§7) inherits the same contradiction until it is.

---

## The treatment matrix

**22. `Grid` — a declared, effective-dated, sparse, cohort-scoped, diffable
N-dimensional table with proven coverage and a generic-kernel evaluation.**
*§5.4's hard part ("authored in a spreadsheet by analysts... attributable
cell-by-cell to an outcome eighteen months later"), §6, acceptance criteria 1
and 11. UGLY — the largest deviation in the sketch, and `matrix/grid.py` says
so directly.*
Doc 03 §4's `Table` sketch is four lines — a 1-D dense array with a present
mask — labelled the "lowest-confidence part" of the document. Nothing in it
survives contact with a 5,376-cell artefact that is 41% sparse, 12% empty,
authored in ~180 rows by non-engineers, held in three simultaneous
cohort-scoped versions, overlaid by a separately-approved stack, and
attributed cell-by-cell eighteen months later. `Grid` replaces it with six
properties, each forced by a spec line: sparse authoring under a declared
most-specific-wins priority (two equal-specificity rows that overlap are a
build error naming both, never resolved by file order); coverage *proven* by
`Grid.validate()` expanding the full key space before anything runs, not
merely tested against a corpus; unused cells *declared* per row
(`traffic`/`sparse`/`none`) so the monthly exercise report catches a
declaration that turns out wrong either way; versions that *coexist*,
resolved by `(decision_date, cohort_code)`, so three simultaneous cohort
versions are registrations, not deployments; a *generic kernel*
(`values[linear_index(keys)]`), because doc 08 §3.4's own test — "can one
compiled loop evaluate every instance, supplied as arrays?" — passes, which
is what makes a matrix release a value change with no compile; and a diff
(`Grid.diff(v213, v214)`) that is the actual reviewable artefact, the only
thing in this project a non-engineer signs.

---

## Overlays: a fourth change class

**23. A registered-surface-only overlay target, with no wildcard and no
dotted path.** *§5.4 ("overlays may never weaken a statutory protection...
enforced, not documented and trusted"). EXT — this is what doc 08 §2's three
change classes have no row for.*
Doc 08 §2 gives values, interiors and skeleton, and none of them is "an
approved, effective-dated, ordered, individually-attributable adjustment
layered non-destructively over a value another module already produced,
under a faster and separately-scoped approval path". `overlays/surfaces.py`
is that fourth class made concrete: an overlay resolves against a registry of
`OverlaySurface` ids, and a target that isn't registered fails at *authoring*
time with the owner's name in the message, not at run time. Every statutory
parameter in the project is simply never registered — the permission boundary
is one file a person can read, which is a stronger property than a runtime
check could ever be, because there is no code path in which the check could
be bypassed by a bug.

**24. Directional combiners, and an authoring-time guard that records its own
rejections.** *§5.4 ("may only ever be more conservative than a statutory
rule, never less"; "the attempted definition is itself recorded"), §5.5,
§5.12 (mid-experiment stack reorder). EXT.*
Where a surface allows movement in only one direction, the surface declares
the arithmetic once — `TIGHTEN_ONLY(direction="increase")`,
`ORDERED_CLAMPED(lo, hi)`, `SUPPRESS_THEN_SUBSTITUTE` — so `max()` rather than
assignment makes the asymmetry structural, not a convention a reviewer has to
check on every document. Doc 03 has composition order (§8) but no notion of a
combiner *attached to a surface* that bounds every future overlay regardless
of who approves it. Above that sits `@overlay_guard(stage="authoring")`,
which runs before a document is admitted at all and — unlike a params
validator, which only rejects a *field* — persists its own rejection as an
auditable fact: `register.json`'s `rejected_attempts` block is what that
produces, and it is the difference between "would have failed validation" and
"we have a record that someone tried".

---

## Params without function pointers

**25. `Choice` — a registered-step selector, switched by an enum param.**
*§5.9 ("ranking basis is a parameter, not a constant... switchable without a
release"). EXT.*
The obvious implementation is a config field holding a function pointer, and
doc 08 §1.1 forbids exactly that (the pointer/reference distinction). `Choice(
param="ranking_basis", options={1: marginal_value_rank_key, 2:
policy_priority_rank_key})` keeps both bases in the graph and in `lineage()`
and `render()` whichever one is selected, checks the ids at validation with a
did-you-mean, and records which one ran (`ranking_basis_code`) as part of the
evidence. Doc 03 has no combinator for "one of several registered,
fully-typed alternatives, selected by a value" — only `Branch`, which
`Choice` is close to but distinct from: a `Branch` arm typically diverges on
record-level data, where `Choice` diverges on a project-level policy setting
that changes rarely and must be auditable when it does.

---

## The same rules, twice a day, once a call

**26. `rerun()` — the same pipeline re-invoked at a later knowledge cutoff,
diffed against the prior derivation; and declared pipeline slicing for the
real-time entry point.** *§5.12 ("an account that paid at 11:00 must not be
telephoned at 14:00 by a list built at 05:00"; "how do batch and real-time
share one implementation"), acceptance criterion 4, NFR latency. EXT for the
rerun; SAT for the slicing.*
`intraday = rerun(daily, trigger="material_change",
emits_diff_against="morning_assignment", withdrawal_allowed_while="queued")`
makes the 13:30 re-run the *same* `decision_date` at a later
`knowledge_cutoff` — a second derivation of the same day, not a new decision —
whose output is a three-way diff (`confirmed`/`withdrawn`/`added`) against the
morning's, not a replacement record. Doc 03 has `apply()` and `score()`; it
has no third shape for "re-run later and reconcile against what already
shipped", needed because a dispatched treatment cannot be recalled and a
queued one can. `live = daily.without(population_dependent=True).for_record()`
needs nothing new *given* #14: it is doc 03 §5's module composition working
exactly as designed once `@population_dependent` exists to mark what to drop
— which is itself the argument for #14, since without that marker this would
be a second, hand-maintained pipeline.

---

## Closing: artefact identity and structural separation

**27. Independent artefact versions recorded as a tuple; and cohorts kept
apart from overlays by structural fact, not convention.** *§5.2, §6
("matrix_version 214 with overlay set 87 and ... 91 are different
configurations and must be recorded as such"), §5.12 ("a design that
expresses a challenger as an overlay scoped to some accounts will produce
results nobody can interpret"). EXT for the versions; SAT for the
separation.*
Doc 03 §3.3's version chain gives exact producer attribution for *values*
crossing scope boundaries inside one compiled graph. This project carries at
least five independent *artefact* identities on every decision —
`matrix_version`, `adjustment_set_version`, `panel_version`,
`path_transitions.version`, `intervals.version` — each on its own release
cadence and owner, recorded as a **tuple**, not merged into one "config
version"; nothing in doc 03/04 currently says a decision record must carry N
independently-versioned artefact identities side by side. Separately,
`cohorts/assignment.py`'s own comment names why cohorts cannot collapse into
overlays despite looking similar: different artefacts (an `Experiment` schema
has no `magnitude`/`stack_position`; an overlay has no `arms`), different
reach (a challenger selects a matrix *version* via `Grid`'s
`cohort_scoped=True`; an overlay modifies a *value* the matrix produced), and
different neutrality obligations (`cohort_scoped: false` on an overlay
triggers `overlays/guard.py`'s demonstrable-equality check; `cohort_scoped:
true` marks every experiment it touches `arms_not_comparable`). No new
primitive was needed for this — `Grid`'s and `OverlayStack`'s own
`cohort_scoped` flags (#22, #23) already exist for other reasons — but the
separation holds only because neither schema can express the other's, and a
framework that unified them "for convenience" would let an overlay's scope
silently become an experiment split.
