# FRAMEWORK-DEMANDS.md

Forty-two demands this project places on `decider2`, each traced to the spec section that
forced it, each citing the file in this tree where the sketch answered it, and each marked
exactly one of:

- **satisfied by doc 03** — the current authoring-API proposal (docs 02–08 collectively;
doc 03 is the load-bearing one) already provides this, sometimes via a specific mechanism
in doc 02, 04, 07 or 08 named in the text.
- **needs extension** — doc 03's shape is right in spirit but does not cover this case,
and the sketch had to invent the missing piece.
- **doc 03 would make this ugly — here is what the sketch did instead** — doc 03's
proposed mechanism, applied literally, would produce something worse than what this project
needs, so a genuinely different shape was used.

Numbers are load-bearing: `ruleset/__init__.py`, `events/catalogue.py`, `decision/overlays.py`
and eleven other files in this tree already cite specific demand numbers inline (`grep -rn
FRAMEWORK-DEMANDS.md .` from the project root finds all of them). Do not renumber this list.

Five facts about this project explain why the list is this long and this insistent: it
authors 400–700 flat rules through non-engineers on a ten-minute clock (§10.1); it must
capture *every* rule that fired, not the winner, as required output at 3 500 events/second
with no sampling (§5.10, acceptance criterion 3); one rule set runs over twelve event types
with different fields (§4.1); the same rule definitions must produce bit-identical answers
in real time and in a 90-day, 420-million-event backtest (§5.17, acceptance criterion 6);
and everything above has to survive intact when the threshold that decided it was itself
moved by a two-minute, self-expiring overlay that nobody edited the rule to apply (§6.5).

---

## A. The rule set as governed data (1–9)

**1. A flat rule set needs a document artefact, not 635 `@step` functions.** 635
analyst-tuned, four-eyes-approved, effective-dated rules cannot be 635 Python files nobody
with the authority to change them is allowed to commit (spec §6.1, §10.1, Q1). File:
`fraud_interdiction/ruleset/__init__.py` ("Why not 635 `@step` functions"). **Verdict:
satisfied by doc 03** — via doc 08 §3's `ruleset` kind: a data-shaped module that
declares its interface (`reads`/`writes`) in Python and takes its body from a validated,
closed-vocabulary document. This is the correct answer to Q1, adopted unmodified for the
rule population itself.

**2. Rule attributes read by the resolver must never be reachable from a predicate.**
`action_code`, `severity`, `priority`, `queue`, `status`, `effective_from/to` are values a
resolver reads; a rule's `evaluate:` block must never see them (spec §6.1 table). File:
`ruleset/__init__.py` `RULE_ATTRIBUTES`. **Verdict: needs extension** — doc 03 §4's
params/structure split governs *tunables* referenced by a predicate; it has no concept of
a second class of per-rule field a predicate may never reference at all. `attributes=` on
`ruleset(...)` names this.

**3. The compile key must be a projection of the interior document, not the whole
file.** A typo fix in `describe.internal` must not cost a compile and an approval cycle
(spec §6.2 — 85% of changes are threshold-only; §10.1's ten-minute budget). File:
`ruleset/__init__.py` `shape_projection=("evaluate", "applies_to.event_types", "on_absent",
"tunables.*.unit")`. **Verdict: needs extension** — doc 08 §2/§3 treats "any interior
change recompiles" as a binary; it does not narrow the compile key to the fields that
actually affect codegen, and one rule document mixes both kinds of field.

**4. One input schema cannot express twelve event types differing by thirty fields.**
`three_d_secure_outcome` exists on event type 111 only; `amount_zar_cents` on six of
twelve (spec §4.1, §11.5's new type 317). File: `fraud_interdiction/events/catalogue.py`
(`VariantSet`, `FieldCatalogue`). **Verdict: doc 03 would make this ugly** — doc 07 §5's
`build --schema` takes *one* schema. Forcing twelve variants through it means either 210
nullable columns per type or twelve near-duplicate pipelines; the sketch makes the schema
variant-aware, keyed by `event_type_code`.

**5. That schema must be effective-dated per field, not just per variant.** A scheme
field addition is absent on every historical event; a backtest across it must know
that, not treat it as null (spec §11.6, Q13). File: `events/field_catalogue.csv`
(`available_from`/`available_to`), `catalogue.py assert_rule_fields_available`. **Verdict:
needs extension** of #4 — same catalogue, with a temporal axis doc 07 §5's schema has
no notion of: a column can now not-exist-yet, distinct from being null.

**6. Doc 03's three null tiers are one bit wide; this domain needs four states and
a watermark.** Stale is not missing, missing is not zero, and a login structurally
has no beneficiary velocity — three different things (spec §4.3, §5.4). File:
`enrichment/velocity.py`, `contracts/feature_vector.json` `state_encoding`. **Verdict: doc
03 would make this ugly** — doc 03 §1 explicitly rejects a `.value`/`.valid` wrapper as
exposing an unchecked accessor. The sketch reintroduces one (`Observed[T]`) but contains
the cost: it is scoped to the ~30 hand-written enrichment steps only, and no rule document
ever mentions `Observed` — a rule declares `on_absent:` and codegen does the rest.

**7. A frame-tier primitive is needed to unpack a fixed-width bitset against a rule
catalogue.** Per-minute firing counts by rule, over 140M events/month, without 635
hand-written shift-and-mask expressions that silently stop matching the catalogue (spec §5.15,
§9.3). File: `pipelines/backtest.py` `rule_level_aggregates`, `bitset_explode`. **Verdict:
needs extension** — doc 02 §1's frame tier ships `join`/`aggregate`/`filter`; nothing
turns N `uint64` columns into `(rule_index, fired)` long rows against a pinned generation's
catalogue.

**8. A derived-and-not-stored set needs a checked digest, not just trust.** The applicable
population must reconstruct from 28 bytes without storing 635 identifiers per record —
but a silent reconstruction bug is a silently wrong audit answer (spec §5.8's "the awkward
one"). File: `decision/applicability.py` `applicable_digest`. **Verdict: needs extension**
— "derive at replay, verify against a stored digest" is a general pattern worth a named
framework helper, not one project's hand-rolled 64-bit hash.

**9. Shadow isolation must be provable by lineage, at import time, with no data.**
"No shadow rule has ever changed an outcome" cannot be a sampled property (spec
§5.11, §9.3, acceptance criterion 4, Q9). File: `tests/test_shadow_isolation.py`
`test_action_code_lineage_excludes_shadow`. **Verdict: satisfied by doc 03** — via
doc 03 §9's `pipeline.lineage(name).sources`, exactly as proposed. The only addition is
disjoint output namespaces per population (`LiveRules.writes` vs `ShadowRules.writes`,
doc 03 §3.1's no-overwrite rule extended across populations) — convention, not a gap.

---

## B. Overlays, concurrency, and floors that don't short-circuit (10–14)

**10. Overlays make a threshold's effective value vary per record, which doc 02 §4 forbids
outright for params.** An overlay's scope is segments, channels and event types — the value
a rule reads is a function of the record (spec §5.9, §6.5, Q5). File: `decision/overlays.py`
header. **Verdict: doc 03 would make this ugly** — the sketch does not make params vary
per record; doc 02 §4's fixed-NamedTuple-per-invocation guarantee stays intact. It adds
a fourth mechanism outside values/interiors/skeleton entirely: a per-event **gain vector**
(5 families × 8 units, 320 bytes) built once per event from a governed `AdjustmentRegister`
document, multiplied into a tunable at the point a predicate reads it. This is probably
the single largest new surface this project asks the framework to own.

**11. Rule-scoped overlays need a keyed, invalidate-on-generation cache, not a per-event
or per-rule rebuild.** Scope-restriction and severity-shift overlays can name individual
rules — a 635-wide vector that cannot be rebuilt at 12 000 events/second (spec
§6.5's scope-restriction kind). File: `overlays.py` `rule_scoped_overlay_vectors` under
`@keyed_materialisation`. **Verdict: needs extension** — nothing in doc 02/03/08 proposes
a declared, deterministic, capacity-bounded memoisation combinator keyed on a projection
of inputs and invalidated wholesale on a generation change. Genuinely new machinery.

**12. Independent enrichment sources must be issued concurrently, not sequenced.** The
model score's 8ms deadline must overlap the other lookups or the 25ms budget fails (spec
§5.5, §8). File: `enrichment/__init__.py` `fan(Counterparty, DeviceSession, Merchant,
Velocity, ModelScore)`. **Verdict: doc 03 would make this ugly** — doc 03 §8.1's `|`
is sequence, and doc 02 §1.2 explicitly rejects inferred fusion or optimisation. `fan()`
is new: concurrency, not codegen fusion. Flagged honestly in this project's own README
§9 as the fifth execution combinator (`|`, `fuse`, `parallel`, `Branch`/`Loop`, `fan`)
— a taxonomy problem for the framework, not this project, to resolve.

**13. A 90-day batch must invoke a fixed-type kernel once per effective-dated generation,
not once for the whole window.** Production ran against however many rule set and overlay
generations were live across 90 days; picking one generation for all of it is wrong, and a
per-record params bundle would break doc 02 §4's fixed-type guarantee (spec §5.17, §9.2,
acceptance criterion 6). File: `pipelines/backtest.py` `PartitionByEffective`. **Verdict:
needs extension** — doc 02 §5's declared frame-tier operations (join/aggregate/filter)
have no temporal-generation partition-and-dispatch operator. Without one, "backtest across
effective-dated params" is dishonest or bespoke on every project with this shape.

**14. A floor over the resolved action must not be a `Branch`.** Full evaluation of the live
and shadow set must continue after a hard block, not stop — the analyst tuning the mule
family needs to know what else would have caught it (spec §5.7: "these short-circuit the
action... but not the evidence"). File: `decision/hard_blocks.py`. **Verdict: doc 03 would
make this ugly** — doc 03 §8.2's `Branch` compiles to a real branch where only the taken
arm executes; here that would skip exactly the evaluation the dispute pack needs. The sketch
feeds the gate as an ordinary value into the resolver's precedence document instead (see #23).

---

## C. Absence, counterfactuals, and temporal tables (15–19)

**15. A rule needs a third truth value — unevaluable — driven by a declared per-feature
policy, distinct from `Observed`'s absence.** True/false/unevaluable, chosen per referenced
feature by the rule's author while she is thinking about fraud, not watermarks (spec §5.4,
§5.10, Q8). File: `ruleset/rules/mule_scam.rules.yaml` `on_absent:` blocks. **Verdict:
needs extension** — doc 03 §1's null tiers describe how a *step* handles a missing
input; nothing describes how a *declarative rule node* should let its author choose among
named codegen strategies (`evaluate_false`/`last_good`/`suppress`) per referenced feature,
checked as closed vocabulary at validation time.

**16. `.at()` must relabel outputs, not only inputs.** The counterfactual resolution has to
be the *same* pydantic object, relabelled onto the base firing set, or it can drift from the
thing it counterfacts (spec §5.12, §6.5 property 2, §9.1). File: `decision/resolve.py`
`ResolveBase = Resolve.at(..., outputs={...})`. **Verdict: needs extension** — doc
03 §5.2 documents `.at(inputs=...)` only. Symmetric output relabelling is what lets
"what would we have decided without the overlay stack" be one line rather than a second,
divergence-prone implementation.

**17. A second table kind is needed: `(key, instant) -> row`, stored as intervals.** A
2M-row watchlist refreshing 24×/day, retained 7 years, cannot be 61 320 snapshot copies
(spec §5.3, §6.4, Q11). File: `tables/definitions.py` `TEMPORAL = IntervalStore(...)`,
`enrichment/counterparty.py`. **Verdict: doc 03 would make this ugly** — doc 03's "Tables
(keyed lookups)" section is explicitly its lowest-confidence part and sketches only `key ->
row` at one instant. Bolting membership-as-at-an-instant onto that means fake snapshots or
a bespoke store outside the framework; `temporal_table` needs to be a first-class second kind.

**18. Backing store must be chosen from a table's declared row count and cadence, not by
the author.** The same `[key].attribute` expression has to read a 1 024-row dense array
and a 3.2M-row hash index without the step knowing which (spec §6.4 — sixteen tables, 9
rows to 3.2M rows, one interface). File: `tables/definitions.py` `backing="dense_sorted"` /
`backing="hash_index"`, `enrichment/merchant.py`. **Verdict: needs extension** — doc 03's
`Table` sketch is unconditionally "dense array + present mask underneath." A framework
serving sixteen tables across four orders of magnitude needs the declaration to select
the representation.

**19. A table needs a declared `diffable`, allowed to degrade honestly to
`"summary_only"`.** A 3.2M-row daily diff is not a reviewable change list (spec §6.4:
"every table requires... diffability"). File: `tables/definitions.py` `MERCHANT_REPUTATION
diffable="summary_only"`. **Verdict: needs extension** — a governance property of the
table declaration, parallel to `contract=` on a module (doc 03 §5.1), that doc 03/08
don't specify for tables at all.

---

## D. Populations, generations, and the compile/audit surface (20–25, 38–39)

**20. One rule set declaration needs three named, disjointly-compiled populations.**
`live`/`shadow`/`retired` are a construction-time property, not a runtime flag — that is what
makes shadow isolation structural rather than conventional (spec §5.8, §5.11, §9.2). File:
`ruleset/__init__.py` `LiveRules`/`ShadowRules`/`RetiredRules = ruleset(population=...,
...)`. **Verdict: needs extension** — doc 08 §3's one worked `ruleset(...)` example
has one population. `population=` as a construction-time discriminator producing genuinely
separate compiled kernels and write-namespaces from one Python declaration is new.

**21. Retired rules must compile only into pinned historical generations, addressable by
event timestamp, indefinitely.** 1 360 retired rules must still replay against a 2024 event
in 2031, without living in the live kernel (spec §9.2, acceptance criterion 9, change
scenario 11). File: `ruleset/rules/retired/2024-h2.rules.yaml` `sunset_policy`. **Verdict:
needs extension** of #20 — doc 08 §4's lifecycle model holds one active and at most one
staging generation; it says nothing about an ever-growing archive of historical generations
retained on a different clock (`retain_replayable_days: 2555`) than what's compiled into
the live kernel (`retain_in_live_kernel_days: 0`).

**22. `score()` and `apply()` must agree with each other, not only each with itself.**
The daily equivalence job (acceptance criterion 6) replays through whichever entry point
production used, and the two take different default fusion paths (spec §5.17, §9.2). File:
`pipelines/backtest.py` `daily_equivalence_check`, `tests/test_replay_equivalence.py`
`test_entry_points_agree`. **Verdict: needs extension** — doc 02 §1.2 deliberately
makes `score()` fuse maximally and `apply()` never fuse by default. Doc 03 §11's
`assert_modes_agree` tests interpreted≡stepped≡fused *within* one entry point and is
silent on the two entry points agreeing with each other — a fourth rung the ladder needs.

**23. Four unrelated governance artefacts — action precedence, degraded-mode selection,
hard-block gates, queue/challenge routing — all want the same "N rows × M uniform-operator
columns" generic-kernel shape.** Each is owned by a different person on a different
cadence (spec §5.6, §5.7, §5.12, §5.13, Q7). Files: `decision/resolve.py PRECEDENCE`,
`decision/hard_blocks.py HARD_BLOCK_TABLE`, `enrichment/completeness.py DEGRADED_MODE_TABLE`,
`decision/dispatch.py CHALLENGE_MATRIX`/`QUEUE_MATRIX`. **Verdict: satisfied by doc 03**
— via doc 08 §3.4's generic-kernel test. Four independent uses of an already-proposed
mechanism is evidence it generalises; one gap worth naming is that doc 08 §3.4 only works
out `decision_table` and `scorecard`, not a lexicographic-argmax-with-override-stages or
a three-state cell grid (`permitted`/`not_permitted`/`not_configured`).

**24. The audit record needs the toolchain that produced the compiled artefact, not only
its hash.** A 2031 replay that rebuilds from pinned source on a different numba/LLVM
is not obviously the same computation (spec §9.2, acceptance criterion 5's 540-day
window — the regulatory one is 7 years). File: `artefacts/decision-record.example.json`
`generation.toolchain`. **Verdict: needs extension** — doc 08 §8's audit-record table
has "compiled artefact id" and no toolchain row. Over these timeframes that's not enough.

**25. A rule needs two version counters so the change class is visible in the artefact,
not only inferable from a diff.** `version` and `shape_version` answer Q4's "is that
distinction visible in the artefact or only in a review process?" directly (spec Q4,
§6.2). File: `ruleset/rules/mule_scam.rules.yaml` header, `production.params.json` per-entry
`change_class`. **Verdict: needs extension** — doc 08 §2's three change classes classify
a *document*, not a *field within one document*; a rule interior mixes threshold-only and
shape-affecting edits in one YAML. Named honestly as unresolved in this project's README
§9: a framework that derived the change class from the diff would be better, and the
sketch could not see how to make that legible in a YAML review.

**38. Compile-unit assignment must be by content hash, never by position.** Adding the
522nd rule must recompile one block, not renumber and recompile all sixteen (spec Q15,
acceptance criterion 15). File: `ruleset/__init__.py` `blocks=stable_blocks(by="rule_id",
count=16)`. **Verdict: needs extension** — doc 08 §4.2's cache-key warning about non-unique
topological sorts is about *step* ordering inside one kernel. Nothing proposes explicit
content-hash sharding for splitting one large ruleset interior into N independently-cacheable
compile units; without it, rule-set growth is quadratic in the wrong place.

**39. One kernel must emit two evaluations per overlay-touched rule — effective and base
— sharing feature-vector reads.** The counterfactual must not cost a second full pass over
635 rules (spec §5.10). File: `ruleset/__init__.py` `emit_base_pass=True`. **Verdict: needs
extension** — a codegen detail specific to the `ruleset` kind that doc 08 §3.4 doesn't
anticipate, because doc 08's `ruleset` sketch has no overlay concept to counterfactual
against; it falls straight out of #10 existing at all.

---

## E. Params documents, contracts, and validation UX (26–32)

**26. Params must be effective-dated per entry, not one flat bundle per generation.**
"What was MS-0208's amount threshold at 14:22 on 3 March" must answer from the params
document alone (spec §9.1 table, §6.2). File: `config/interdict/production.params.json`
`rules.MS-0208.entries[]`. **Verdict: doc 03 would make this ugly** — doc 03 §4's
params model is one validated bundle per invocation with no date axis. The sketch keeps
the fixed-NamedTuple-per-invocation guarantee but resolves *which* entry populates that
bundle by date against the generation in force — a standard shape doc 03/08 don't name.

**27. Approval must be able to route on which field moved.** A threshold takes four-eyes in
six minutes; `critical` and `overlay_exempt` take two named people; an overlay takes a duty
officer plus one (spec §6.1 table, acceptance criteria 1–2). File: `production.params.json`
per-entry `approval` blocks, `mule_scam.rules.yaml approval:`. **Verdict: satisfied by
doc 03** — explicitly *not* framework territory per doc 04 §2.1 ("who may activate a
staged pipeline... the caller's policy"), correctly so. The document just needs room for
structured approval metadata, which an ordinary object field already gives it.

**28. A pipeline-level contract must fail the build if a required record field isn't produced
by anything.** "A record that fails to persist is an incident, not a dropped metric" has to be
enforceable statically (spec §5.15, acceptance criterion 3). File: `pipelines/interdict.py`
`record_contract(Interdiction, "contracts/decision_record.json")`. **Verdict: needs extension**
— doc 03 §5.1's `contract=` freezes one *module's* interface for exact-match semver
purposes. This needs a *pipeline-level*, superset-of check against an externally-owned
schema — a different comparison than a module contract makes.

**29. The synchronous response to an external system must be a declared, reviewed projection,
never the record itself.** "A merchant acquirer does not learn which of the Bank's rules
stopped a card" (spec §7). File: `pipelines/interdict.py` `SYNCHRONOUS_RESPONSE`. **Verdict:
satisfied by doc 03** — an ordinary named subset over a pydantic-described pipeline (doc
03 §5.1's materialised interface). Included to record that this boundary needs nothing new.

**30. A derived quantity must be a registered step referenced by id, and the validator's
error should hand the analyst a stub, not a dead end.** 3% of rules need a ratio; doc
08 §3.2 already settles that it's code, not an expression string (spec §5.10). File:
`features/derived.py` `@feature(id=...)`, `ruleset/__init__.py validate_documents`. **Verdict:
satisfied by doc 03** — via doc 08 §3.2, adopted unmodified. The one addition
(closest-registered-feature-plus-generated-stub in the error message) is a UX refinement
of an existing did-you-mean pattern (doc 02 §2.2), and it is what keeps an analyst under
attack at 19:40 on a Friday from inlining a literal instead of waiting for a deploy.

**31. Feature-availability validation must intersect a rule's whole effective window
against a field's whole availability window, not just check at authoring time.** A rule
authored before a field existed on 46 of its own days must be caught before deployment (spec
§11.6, §4.1). File: `events/catalogue.py assert_rule_fields_available`. **Verdict: needs
extension** of #4/#5 — same catalogue, but the check itself (interval-intersects-interval
between a rule's effective window and a field's availability window) is a validation shape
doc 03/08 don't describe anywhere.

**32. Degraded-mode combinations must be enumerated in a document, never derived from
source states.** Splitting the model into two doubles the presence states from two to four;
deriving the four from the two source flags is how a scam-model incident silently switches
off the mule family (spec §11.7, §5.6). File: `config/interdict/degraded_modes.json
model_presence_states`. **Verdict: satisfied by doc 03** — via doc 08 §3.4's generic-kernel
`decision_table` (see #23). A modelling discipline, not a framework gap, worth stating
because the wrong instinct here is a natural one.

---

## F. Automated changes, disagreement, and vocabulary (33–37)

**33. A monitor-written params value must land in the audit record distinguishably from
a human-authored one.** A rule's own fire-rate circuit breaker writes a params field
with no human review, and it is outcome-affecting (spec §6.1, acceptance criterion
14). File: `production.params.json circuit_breaker_state`, `decision/applicability.py
circuit_open_bits`. **Verdict: needs extension** — doc 04 §5.2's audit record implicitly
assumes resolved params came from a reviewed document. Nothing currently marks "changed
by an automated control" as distinct from "changed by a person under four-eyes."

**34. Circuit-trip attribution must name what else was in force, not just that the rule
tripped.** If an overlay pushed the rule past its own fire rate, demoting the rule is the wrong
remedy (spec §6.1: "must attribute the trip correctly"). File: `decision/applicability.py
circuit_trip_attribution`. **Verdict: needs extension** of #10/#11 — once overlays exist,
any framework-level fire-rate/circuit-breaker primitive needs "what else was in force"
as a required input, not an afterthought.

**35. A resolver needs a governance-exception output for the case its own tie-break resolves
deterministically but not correctly.** Two critical rules firing with different actions
means two owners made incompatible assumptions, and the tie-break papering over that has
to be visible and routed (spec §5.12 stage 4, §11.12, Q7). File: `decision/resolve.py
governance_exception`. **Verdict: satisfied by doc 03** — an ordinary step output plus
a tap (doc 03 §7). The interesting part is a modelling discipline, not new machinery.

**36. Wording must store both the internal reason set and the rendered external text,
with the mapping between them.** A complaint is about what was said; a representment
is about what was true (spec §5.14). File: `decision/dispatch.py client_wording`,
`artefacts/decision-record.example.json client_communication`. **Verdict: satisfied by
doc 03** — an ordinary multi-output step (doc 03 §1). No framework gap; only a reminder
that the *rendered* text must be the thing stored.

**37. Where does invented vocabulary belong — framework or project?** Fifteen names
(`rule_id`, `action_code`, `fired_rule_ids`, `Observed`, `temporal_table`, `precedence`,
`population`, `keyed_materialisation`, `fan`, `stable_blocks`, ...) are declared locally
because nothing publishes them, and the answer changes depending on whether project
08 arrives with its own flat rule set and its own overlays (spec Q16, §4.4). File:
`README.md` §10 Q16 row. **Verdict: needs extension** — a `core` package question
doc 02 §6's package layout gestures at without settling. `Observed`, `temporal_table`,
`precedence`, `keyed_materialisation` and `fan` read as framework-general; `rule_id`,
`action_code`, `fired_rule_ids`, `family_index` read as irreducibly this domain's, and no
document draws that line.

---

## G. Backtest honesty and residual risk (40–42)

**40. The one place in this design whose correctness argument is a cache-invalidation
argument, not a static one, deserves a framework home.** `rule_scoped_overlay_vectors`
(#11) is correct because it's a pure function of its key — but "correct because the
cache key is complete" is a weaker sentence than everything else in this design leans on,
and it is the first thing this project's own README (§9) says it would load-test. File:
`decision/overlays.py`. **Verdict: needs extension** — restated separately from #11
because it is the sketch's own top self-identified risk and belongs in front of framework
maintainers as a determinism obligation to prove once, centrally, rather than once per
project reaching for `keyed_materialisation`.

**41. Retirement and overlay-lapse must be provable by absence from a cheap aggregate,
not by scanning the record store.** "Prove `MS-0117` stopped affecting outcomes when
you say it did" and "prove `ADJ-0042` stopped applying on its expiry date" are both
any-time questions over 140M records/month (spec §9.3, acceptance criteria 8–9). File:
`tests/test_replay_equivalence.py test_retired_rule_has_no_effect_after_its_instant`,
`overlays/register.yaml lapsed_recently`. **Verdict: needs extension** of #7 — the same
`bitset_explode` + per-minute rollup primitive, cited separately because it answers a
distinct acceptance criterion (expiry evidence) from the one #7 was justified against
(firing-set output volume).

**42. A backtest result must state which of two baselines and which overlay mode produced it.**
"A result that does not state which mode produced it is not admissible at the Committee"
(spec §5.17, acceptance criterion 7, change scenario 14). File: `pipelines/backtest.py
backtest(baseline=..., overlay_mode=...)`. **Verdict: satisfied by doc 03** — an ordinary
keyword-argument API, recorded in the run metadata doc 04 §5.2 already asks for. No new
machinery; only a reminder that a backtest's *inputs* are as auditable as its outputs.

---

## The one thing to take away

Thirty of these forty-two demands are ordinary — a projection, a tap, a document with
room for an approval block. The framework work concentrates in three places: **overlays as
a fourth, per-record-varying mechanism** (#10, #11, #34, #39, #40) sitting entirely outside
doc 02 §4's params/structure binary; **a second table kind** for membership-as-at-an-instant
(#17, #18, #19); and **a variant-and-effective-dated schema** replacing doc 07 §5's single
flat input schema (#4, #5, #31). If `decider2` picks up one thing from this sketch, it
should be the first: adjustments are not params with extra steps, and pretending otherwise
is what would have made this project's central requirement — an overlay that never edits
its base and always expires (§6.5) — impossible to build honestly.

---

## Fix applied to an existing file

`README.md`, §3, the paragraph on **Applicability**, read: "...from six recorded scalars:
event type, segment bits, timestamp, degraded mode, and the suspension and circuit-breaker
masks." That list does not match the six scalars `applicable_bits` actually takes and
records (`decision/applicability.py`'s own docstring: event type, segment bits, timestamp,
degraded mode, **rule set version, and adjustment stack version**), and it does not match
`artefacts/decision-record.example.json`'s `applicability.inputs` object, which carries
exactly those six fields. Suspension and circuit-breaker masks are themselves *derived*
from degraded-mode code and the adjustment stack, not independently recorded, so naming them
as two of the "six recorded scalars" was an outright inconsistency rather than a design
choice. Corrected in place to read "...degraded mode, rule set version, and adjustment
stack version" — no other change to that file.
