# FRAMEWORK-DEMANDS — 05 Business credit granting with nested entities

Twenty-two numbered demands this project makes of `decider2`. Each names the
spec section that forced it and is marked:

- **[SATISFIED]** — doc 03 (or 02/04/07/08) already does this, unchanged.
- **[EXTENSION]** — the shape doc 03 already has is right; something specific
  is missing.
- **[UGLY]** — doc 03's existing answer would produce something not fit to
  put in front of a credit committee. What the sketch did instead is stated.

Seven of twenty-two are [UGLY]. The nesting demands (A) come first, because
they are what this project exists to surface — doc 06 O5, "ragged per-record
collections," is deprioritised and unresolved, and every other demand here is
downstream of how A is answered. A short list of what leans on doc 03
**unchanged** closes the document, on purpose: grains reuse ordinary
`decider2` everywhere except the handful of points below.

---

## A. The grain and the fold — first, because they are the point

### D01. A collection needs a declared type: key, parent, capacity, identity, order. [UGLY]

**Forced by:** §4.2 (1..40 entities), §4.4 (0..60 events per entity, 600
worst case), §13 Q1/Q5.

Doc 03's only iteration primitive is `Loop(..., carries=[...],
max_iterations=N)` (§8.3). Nesting two nested `Loop`s over data-length
collections breaks three things at once: static lineage stops at the loop
body, "bound exceeded" has to be hand-coded rather than declared, and nothing
stops the body writing `events[0]`, so ordering independence (§8) becomes a
hope rather than a guarantee. `grains.py` answers with a fifth thing between
step and module: `grain(key=, parent=, identity=, capacity=Capacity(n,
on_exceed=...), order=(...))`. `capacity` is part of the **type** (a 4-entity
and a 40-entity application compile to the same kernel); the **policy** limit
underneath it is an ordinary `param()`, so scenario 3 (depth 3→4 above R5m)
is a values change, not a recompile. `identity` is what attribution resolves
on — ordinals never escape the kernel, answering Q5 ("positional identity
fails the moment an event is removed") directly.

### D02. A fold vocabulary that admits no `first`. [UGLY]

**Forced by:** §5.6's roll-up requirements, §8's shuffle-invariance test
(1 000 applications × 20 shuffles), §13 Q7.

A `Loop` accumulator can write "the worst event so far" exactly as easily as
`best = events[0]` and forget to update it — ordering independence then rests
on discipline. `Gather(Event, into=Entity, **folds)`
(`entities/adverse/gather.py`) restricts every fold to a small,
commutative-associative set — `count`, `sum_of`, `min_of`, `max_of`, `any_of`,
`best_of`, `collect_of` — and `best_of` **requires** a total tie-break ending
in a stable identity. There is no way to write "the first one." Every fold
also carries a **witness** (the child rows that produced it) for free, as a
bitset over ordinal position, never written and never forgettable.
Shuffle-invariance becomes a build-time property of the fold vocabulary; the
randomised test becomes a regression check on it, not the proof of it.

---

## B. More ragged collections than the two headlined, and what bounds them

### D03. The design must generalise past the two collections the spec names. [EXTENSION]

**Forced by:** §4.5 (`security_offered`, 0..12) and §4.5 (1..3 unequal-length
financial periods).

`people/surety.py` sums a 0..12 collection inline as a plain `Gather`, without
a named grain, because it never crosses a grain shift the reader must follow.
`financial/measures.py` gives the financial spread a full fifth grain,
`Period`, because three unequal-length periods need per-period logic before
they roll up. The demand is that `grain()` stay cheap enough to declare for a
collection this small and local — a design that only handles the two
headlined nestings is a special case, not a design.

### D04. Capacity breach is a declared per-grain outcome, not one global answer. [EXTENSION]

**Forced by:** §5.1 ("does not decline... suspends the automated outcome")
versus §5.12 ("the best admissible candidate found so far, flagged").

`Entity.materialise()` on a >40-entity structure must **suspend**
(`structure_unresolved`); `Candidate` exceeding 2 200 must **truncate** and
flag `search_truncated`. `Capacity(n, on_exceed="suspend"|"truncate",
flag=...)` makes this a declaration on the grain, not a convention an author
must apply consistently — both behaviours the spec needs are first-class.

### D05. A witness needs a correct answer above the machine-word capacity it was built on. [EXTENSION]

**Forced by:** §4.2/§4.4 (`Entity`/`Event` capacities 40 and 60), and the
sketch's own §13 Q6 answer that attribution must be **uniform**.

A witness is a bitset over ordinal position, so 40 and 60 both fit one
`int64`. That is a governance bound (40 entities, what the Bank looks
through) landing on a machine word by coincidence. `Candidate` (capacity
2 200) is already outside that regime, and happens not to need a witness
(only a `best_of`). The framework must ship the >64 fallback — re-filtering
the child frame by predicate name, exact but costing a scan — as a real,
tested path: this project already proves "witnesses fit in a word" false at
the edge of its own grain list.

### D06. The equivalence ladder needs a rung for the fold, not only the step. [EXTENSION]

**Forced by:** §8 (bit-for-bit determinism), §13 Q16 (record-at-a-time vs
set-at-a-time producing identical results).

`ClassifyEvent` runs 600 times inside one `score()` call and 6.7M times
inside one `apply()` frame. Doc 02 §3.1's `interpreted ≡ stepped ≡ fused`
ladder proves the **step** is identical across modes; nothing yet proves the
**fold** is — that `count(where=...)` and `sum_of(..., nulls=...)` agree
between a capacity-bounded array and a polars group-by. Doc 03 §1.2's
measured `round()` divergence is exactly the kind of thing a fold accumulator
could reintroduce silently. `assert_modes_agree` needs a fold-shaped
counterpart.

---

## C. One capability, shared across grains, both upward and downward

### D07. A shared capability, parameterised by a value broadcast from a different grain. [EXTENSION]

**Forced by:** §5.5 (`core.adverse_events`'s thresholds vary by the
criticality of the entity two grains up) and §13 Q3 in its entirety.

`vocabulary.py`'s `grains={"core.adverse_events": Event, ...}` pins a library
capability to its consuming grain; doc 03 §5.2 already covers **renaming**.
What it doesn't cover is that the capability needs an input
(`criticality_class`) that is neither its own leaf column nor something it
computes — an ordinary broadcast value from an ancestor grain. The sketch's
answer to Q3 is that this needs no new capability shape, only that "broadcast
from an ancestor grain" be as unremarkable as "leaf column" at the call
site — which name-based wiring already gives, provided the grain declares the
broadcast.

### D08. One step definition, consumed at two grains, re-pinned at the use site. [EXTENSION]

**Forced by:** §5.9's `debt_service_coverage` (Application grain) versus
§5.12's per-candidate DSCR (Candidate grain, 200–600 times per application).

`financial/measures.py` is explicit that duplicating the formula "would
guarantee FV-02 eventually failing" — a real, recorded drift risk, not a
hypothetical. `pricing/price_one.py`'s `candidate_dscr` re-implements the same
arithmetic at Candidate grain, because a step is scoped to one grain by
construction. The demand is that "the same definition, at two grains, chosen
at the use site" become a declared relationship, not two hand-synchronised
functions linked only by a comment.

### D09. The frame tier needs a bounded, declared loop over its own primitives. [EXTENSION]

**Forced by:** §5.1's three-round graph collapse, §13 Q12.

Doc 02 §1 ships `join`, `aggregate`, `filter` and nothing iterative.
Collapsing a disclosed ownership graph to a bounded tree is inherently
iterative: expand a level, drop cycles, drop below-materiality holdings,
repeat, bounded by `Entity.capacity`'s depth dimension.
`structure/resolve.py`'s `FrameLoop(ExpandOneLevel, carries=[...],
max_iterations=3, on_exhausted=...)` is the one frame-tier construct this
project needs beyond doc 02 §1 — narrower than a general frame-tier `Loop`,
because three bounded rounds of join→antijoin→filter should not reopen §8.3's
unbounded-loop problem at a tier where `break`/`continue` don't exist the
same way.

### D10. A grain shift downward: a parent-grain rule outcome becomes new child rows. [UGLY]

**Forced by:** §5.4 — a "material adverse event" disposition is "injected
into that entity's event list at §5.5 with a synthetic event."

`Gather` goes one way, child to parent. `entities/disqualification.py` says
plainly this is "the one that doc 03 has no story for at all."
`emit_into(Event, from_grain=Entity, when=..., fields={...})` is declared as
`Gather`'s literal inverse. Two properties keep it safe: emitted rows are
marked `is_synthetic` with `source_rule_id`, so every downstream count is
reportable with and without them; and the emission is ordered strictly before
`Each(Event, ClassifyEvent)` in the pipeline expression, so the framework can
reject an `emit_into` that would invalidate a `Gather` already computed at
that grain. Without an enforced ordering this is a silent-invalidation hazard;
with one, it's a declared, checkable grain shift.

---

## D. Roll-up expressed as a rule set, not a new kind

### D11. A rule-set kind whose outcome composes by more than "most severe wins." [UGLY]

**Forced by:** §5.3 ("every rule evaluated even after the first fails" —
`collect="all"`, not a waterfall), §5.6 (rules that **modify** a verdict
rather than **give** one — floors, caps, escalation), §13 Q2.

`business/disqualification.py` rules out both existing shapes: `Branch` and a
waterfall of `outcome → outcome` modules both short-circuit, which §5.3
forbids. `verdict(name=, of=, resolve=, collect="all", rules=[fires(pred,
gives=|floors=|caps=|escalates=, reason=, attributes=, quantity=,
names_entity=, unless=)])` is the rule-set kind doc 03 does not have. Three
things beyond `collect="all"` plus a severity order: rules can **modify**
instead of only setting (AE-R-08 floors, AE-R-05 caps the business outcome,
AE-R-10 escalates by one class); the composition order across modifier kinds
is declared (`modifier_order=("gives", "escalates", "floors", "caps")` —
cap-then-escalate and escalate-then-cap disagree and the spec doesn't say
which); and every `fires(...)` carries its own attribution clause rather than
attribution being assembled afterward. Doc 08 §3.4's `ruleset` (codegen,
heterogeneous predicates) is the closest relative — `verdict` needs the same
strategy plus resolve-and-modify semantics `ruleset` doesn't have.

### D12. A rule's outcome is a table lookup keyed on an entity attribute, and an asymmetry in it must be provably deliberate. [EXTENSION]

**Forced by:** §5.4 — a 12×3 disposition matrix is "a policy artefact," and
three rules decline regardless of class, an asymmetry that "must be readable
as deliberate, not as an oversight."

`entities/disqualification.py`'s `fires(e_arod_02_fires,
gives=by_class(DISPOSITION_MATRIX, "E-AROD-02"), class_invariant=True)` does
two things doc 03 lacks: `by_class(table, rule_id)` reads a rule's outcome
from a cell keyed on `criticality_class`, and `class_invariant=True` is a
build-time assertion **against that table** — a later edit softening
E-AROD-12 for peripheral entities fails the build naming the rule. Forty-two
`if`s would hide the asymmetry; a table without the assertion would make a
deliberate exception indistinguishable from a typo.

### D13. Attribution across grains is a declared join of what each grain already recorded, not an assembly. [EXTENSION]

**Forced by:** §7.1 ("`attributing_entity_id`, `attributing_event_ids` — the
output the project exists for"), §9.2, §13 Q4.

Every `verdict(...)` already emits a binding rule id, a fired-rule mask, and
a witness per fold (D02). `evidence/attribution.py` assembles nothing — it
declares `attribution_spine(levels=[(grain, verdict_name, names=child_grain),
...], identity={...}, primary=...)`, a join over those per-grain triples by
key. No level knows about the level below it; a third level (scenario 13)
adds a row to the join, not a rewrite. Doc 03's worked examples never cross
more than one grain, so this composition-across-grain-shifts demand has no
precedent to extend — it needs stating fresh.

---

## E. Pricing as a grain, not a search

### D14. A non-monotone, exhaustive, bounded, fully-recorded search is a grain, not a loop. [UGLY]

**Forced by:** §5.12's four simultaneous requirements — exhaustive; bounded
with a declared truncation outcome; every candidate recorded in order; safe
under a proven non-monotone amount/instalment relationship — and §13 Q10.

`pricing/candidates.py` calls this "the second-biggest departure from doc 03
... after the grain itself": a `Loop` gives "every candidate recorded" only if
the body writes a side record (breaking purity), and "exhaustive" only by a
5 000-application proof after the fact. Declaring the space as `Candidate`, a
derived grain via `Enumerate(Candidate, from_=cross(...), where=[...])`, gets
all four by construction: the fold (`best_of`) sees every candidate; capacity
is the declared bound with a declared truncation outcome; the candidates
*are* a frame, so "why not R2 000 000?" is a row; and monotonicity is never
assumed because nothing traverses the space in an order that could depend on
it. The demand is that enumerate-as-grain be a recognised alternative to
`Loop` for exactly this shape, rather than every bounded search routing
through the same accumulator-loop primitive regardless of fit.

---

## F. Overlays and shadows inside the nesting

### D15. A declared overlay stack needs whole-stack checks at publish time, not per-application checks at run time. [EXTENSION]

**Forced by:** §5.10 — "two overlays occupying the same position with
overlapping scope is a conflict and must be detected when the adjustment set
is published" — and PP-11's named collision between positions 3 and 6.

`grade/overlay_stack.py`'s `on_publish=[overlay.no_scope_overlap_within_position(),
overlay.no_cross_position_double_count(pairs=[(3, 6)],
shared_axis="sector_grouping_code"), overlay.expiry_required(),
overlay.scope_is_reachable()]` are checks over the **declared stack as a
whole**, run once when Credit Committee publishes a set, not per application.
The double-count check cannot be inferred — the framework cannot know two
scope axes denote overlapping populations without being told which position
pairs to compare, and on which shared axis. This is a genuinely new verb on
`overlay.stack(...)`: checks at publication, distinct from build time
(structure) and run time (one application).

### D16. Layering a value over a table must generate its own audit companions. [EXTENSION]

**Forced by:** §5.5 requirements 1–2 — the base value is never edited, and
every classified event records which value was used and whether it was
overlaid, with the overlay's identifier.

`EVENT_THRESHOLDS`'s `overlay=overlay.position(1, scopes=(...))` generates
`material_threshold_base`, `material_threshold_overlay_id` and their
`disqualifying_threshold` counterparts from one declaration. This is the one
place in the project where a forgotten manual `_base` column would look
complete and be silently wrong the moment a committee asked "which threshold,
and was it overlaid?"

### D17. A scope mismatch on an overlay is a build/resolution-time error, checked per element. [EXTENSION]

**Forced by:** §5.7 — "applying a `BUS-PERS-01`-scoped overlay to a juristic
entity is an error, not a silent no-op."

Both scorecard families' overlays run inside one `Each(Entity, ...)`, so the
check that a personal-scorecard overlay never reaches a juristic row must run
**per entity, at resolution** — a fourteen-entity application must not let one
wrong-family overlay through because thirteen others were fine.
`families.py` names this demand directly: `overlay.position(...)`'s scope
must be enforced at the grain it targets, loudly, rather than at the grain the
pipeline happens to be invoked from.

### D18. A shadow's cost must be provable from static lineage, not merely hoped small. [EXTENSION]

**Forced by:** §5.5 AE-C-22 (dispute shadow, always on), §5.8 PP-11 and
§5.6's unadjusted-threshold verdict (also always on), §9.4 (event
counterfactual, on demand), scenario 15.

`Shadow(pipeline, perturb=..., suffix=..., share_prefix=True|on_demand=True)`
re-evaluates a declared sub-pipeline under a declared perturbation, and
appears four times — `UnadjustedSpine`, `DisputeShadow`, `EventCounterfactual`,
`SecurityUpliftShadow` — which is what makes it a construct rather than four
bespoke solutions. The demand beyond "the construct exists": an **always-on**
shadow whose perturbation reaches only a minority must cost proportionally to
that minority, not the batch. `DisputeShadow` reruns the flow's most
expensive stage; computing it unconditionally for every application would
double granting cost to serve the applications with a disputed event.
`share_prefix=True` asks the framework to prove from static lineage which
prefix the perturbation cannot reach and share it; for an application with no
disputed event, that proof must be total. Without it, "the Bank cannot afford
its own policy" — the failure mode named in the code — is the honest
consequence.

---

## G. The tuning surface, disclosure, and what heterogeneity costs

### D19. A param and a one-key table should be the same declaration at two arities. [UGLY]

**Forced by:** §6.1 (the minor-event window is **one** parameter) and
scenario 4 (that window moves for peripheral entities only).

`entities/adverse/flags.py`'s `count_window_months = param(12.0, ...)` is one
scalar today; scenario 4 needs it keyed on `criticality_class` — a one-key
table — which changes the function's **signature**, forcing a recompile and
therefore an engineer, exactly the boundary §6.1's forty-eight parameters
exist to keep Policy off of. The file names this "the sharpest ergonomic
failure in this sketch": a param and a one-key table read almost identically
at the call site and differ only in arity, but `decider2` currently treats
promoting one to the other as structural rather than a values change. The
demand is that this promotion be a config edit.

### D20. Disclosure is a grain-level default with named overrides, not a per-value decision. [EXTENSION]

**Forced by:** §9.2 (the business is entitled to its own reason, generally
not to a third party's record) and §13 Q14.

`disclosure.defaults({Application: BUSINESS, Entity: INTERNAL, Event:
INTERNAL})` plus a short, Compliance-reviewed `disclosure.overrides({...})`
means a new value at the Entity or Event grain is **invisible to the client
until classified**, rather than visible until hidden — the direction doc 04
§5.3's "PII by construction" needs made operational. The demand is that the
default live **on the grain declaration**, because the third-party boundary
here runs exactly along a grain boundary — a coincidence this spec has, that
a general framework must nonetheless be able to declare.

### D21. A policy taxonomy that doesn't line up with the executable rule count needs a joinable table, not a naming convention. [EXTENSION]

**Forced by:** §5.3 — twenty rules, "referred to as fourteen classes... in
policy documents, and the mapping... is itself something the implementation
must keep visible" — and §13 Q13.

`POLICY_CLASS_MAP = table("business_disqualification_policy_class",
key="rule_id", columns=(...), ...)` is an ordinary effective-dated table so
the generated review artefact can group by the vocabulary Policy actually
uses. This generalises: any rule set this size (thirty-four classification
rules, twelve roll-up, twenty disqualification, ten blend — §13 Q13) will
have a policy-facing taxonomy that doesn't divide the executable rules
evenly, and the reviewable artefact (doc 04 §6) must group by such a table,
not only by a rule's own declared identity.

### D22. Heterogeneous per-arm inputs inside a branched collection still cost both column sets on the frame. [UGLY]

**Forced by:** §5.7 — two scorecard families, 38 versus 29 characteristics,
routed by `Branch` nested inside `Each(Entity, ...)`.

`Branch` (§8.2) guarantees only the taken arm **executes** — the
short-circuiting behind doc 01 §2's 7.8× advantage. It says nothing about
**frame width**: `entities/scoring/families.py` states plainly that the
Entity frame must carry all 38 personal and all 29 commercial characteristics
as columns, one side null for every row, because the frame is columnar and a
row is one entity regardless of which arm it will take. No fix is offered,
because none avoids reintroducing heterogeneous per-row schemas, which a
columnar frame tier cannot do cheaply. Stated here rather than smoothed over:
a 40-entity application carries 67 mostly-null characteristic columns instead
of the ~30 either family alone would need.

---

## Marked [SATISFIED] — leaned on unchanged, by design

- **Every per-grain module is an ordinary `module(fn1, fn2, ..., grain=X)`.**
  `grains.py` says it directly: "everything else here is ordinary
  `decider2`." No grain-aware step syntax exists or is needed.
- **Tier-3 `Optional` nulls (§1) apply unchanged at any grain** —
  `unsatisfied_amount`, `months_since_satisfaction`, `turnover_trend` all use
  plain `float | None` regardless of grain.
- **`contract=` frozen interfaces (§5.1) are used unchanged** on
  `classify_event`, `entity_adverse_verdict`, `business_grade` and others.
- **`Branch`'s core guarantee is exactly what the two-family split needed** —
  D22 is a cost on top of it, not a gap in it.
- **A single overlay position targeting one table needed nothing new** — D15
  through D18 are all about the **stack** of eleven positions across four
  grains, not about any one position alone.
