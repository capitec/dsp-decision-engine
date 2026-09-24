# NOTES

## What I publish

Project 11 (business end to end) consumes this project's components directly
(DEPS.md, spec 11 §4.9: "Entity structure resolution... [05] §5.1" through
"Final validation, conditions, committee pack... [05] §5.13"). Entry points,
all under `business_nested/`:

| Capability | Module | Key names |
|---|---|---|
| Structure resolution + criticality | `structure.py` | `resolve_entities(raw_entities, decision_date) -> dict` (pure function, flat parallel lists out); `resolve_structure` (the `frame_step`); `classify_criticality(...)` |
| Adverse event classification | `events.py` | `classify_events(...) -> dict` (pure function); `make_classify_step(overlays, adjustment_set_id)` (builds the `frame_step`); `build_event_threshold_table()` |
| Entity adverse roll-up | `rollup.py` | `rollup_one_entity(events, criticality_class, requested_amount)`; `rollup_entities(...)`; `rollup_step` (the `frame_step`); rule-id constants `AE_R_01`...`AE_R_12` |
| Entity scoring | `scoring.py` | `score_entities(...) -> dict`; `score_entities_step`; `grade_from_pd(pd, segment_code)`; `build_entity_scorecard()`/`build_entity_risk_grade_table()` |
| People blend | `blend.py` | `people_component(...) -> dict`; `people_component_step` |
| Financial assessment | `financial.py` | `build_financial_unit()` |
| Combined grade | `grade.py` | `build_grade_unit()`; `combined_pd_before_overlay(...)`; `build_business_risk_grade_table()` |
| Pricing | `pricing.py` | `build_pricing_unit()` |
| Sole proprietor (regulated) | `sole_proprietor.py` | `assess_sole_proprietor(...)`; `sole_proprietor_step` |
| Single-event counterfactual | `counterfactual.py` | `single_event_counterfactual(record, event_id) -> dict` |
| Partial one-entity re-assessment | `counterfactual.py` | `partial_reassess_entity(record, entity_id, refreshed_entity, original_result) -> dict` |
| Outcome + attribution | `outcome.py` | `business_outcome(...)`; `REASON_REGISTRY`; the `(attributing_entity_id, attributing_event_ids)` pair, "the output the project exists for" (spec 05 §7.1) |
| Shared vocabulary | `vocab.py` | Criticality classes, verdict codes, relationship-type roles, segment codes |
| Full pipeline | `pipeline.py` | `build()` -- the whole assessment as one `dag`, same shape `decider template` produces |

**Consumption pattern**: every ragged/nested stage (`structure`, `events`,
`rollup`, `scoring`) exposes both a pure Python function (no `decider`
machinery, directly callable and unit-testable) and a `decider` step/step-
builder that wraps it -- the same split 00 and 02 use for their own units,
so project 11 can either compose this project's `frame_step`s directly into
a larger `dag`, or call the pure functions from inside its own composition
if the ragged shape doesn't line up. See "Framework friction" 4.4 for why a
nested-`Engine`-inside-a-`frame_step` is how `scoring.py`/`events.py` reuse
`ScorecardConfig`/`DecisionTableConfig` per entity -- project 11 wanting the
same "once per entity" composition at a still-higher level (project 11's
own EP-1, spec 11 §4.9) will hit the identical gap and can reuse this
project's pattern directly.

**Column-naming convention**: every flat, parallel array this project emits
is prefixed by its grain -- `entity_*` (one entry per resolved entity),
`ev_*` (one entry per classified event), `attr_*` (one entry per
attributing (entity, event) pair) -- joined by `entity_id`/`event_id`, never
by position. A consumer must not assume index `i` of `entity_id` lines up
positionally with index `i` of an *unrelated* list without checking they
share the same grain; every list at the same grain is, however, guaranteed
index-aligned and sorted the same way (by `entity_id`, or by
`(entity_id, event_id)`).

---

## 1. What I built

The slice SCOPE.md asks for, over 00's `credit_core` and 02's affordability
assessment:

- **Structure resolution (§5.1, `business_nested/structure.py`).** A natural
  person reached by two disclosed paths is resolved to **one** entity: ownership
  sums across paths, role is the most senior (spec's own ordering), control is
  the disjunction, adverse events attach once (deduplicated by `event_id`),
  every path's count is retained (`entity_path_count`). Bounds are enforced
  (resolved count > 40, or ownership reconciliation outside 90-110%, sets
  `structure_unresolved` -- a referral, never a decline or an approval).
  **Not built**: the actual graph-collapsing algorithm (cycle detection, the
  3-level expansion-with-materiality-floor traversal). The caller supplies
  each disclosed path's own `effective_ownership_pct` already computed; this
  stage does the de-duplication and bounding, not the path-product arithmetic
  over a raw graph. See "What I left out" and "Spec problems".
- **Entity criticality (§5.4's criticality table only, `structure.py`).**
  Critical / significant / peripheral, from ownership, control, required-surety
  and sole-principal flags. The 14-rule entity disqualification matrix (E-AROD)
  and the 20-rule business-level one (B-AROD) are **not built** -- SCOPE.md's
  slice for 05 names only "per-entity criticality", not the disqualification
  matrices; the business decline in this slice comes from the roll-up (§5.6)
  and the people blend (§5.8), not from B-AROD/E-AROD.
- **Adverse event classification (§5.5, `business_nested/events.py`).**
  `core.adverse_events.event_severity_code` (project 00) called directly,
  unmodified, with criticality-class-dependent thresholds from a project-05
  `DecisionTableConfig` (four of the spec's five threshold rows -- judgment,
  default listing, tax non-compliance, litigation -- plus a stand-in for
  "municipal/rental" and a default for every other type). AE-C-22 (disputed
  events downgrade one class, floored at minor, flagged provisional) is
  layered on top as a project-05 wrapper. The threshold overlay (stack
  position 1) is applied to `material_threshold`/`disqualifying_threshold`
  **before** classification, through `core.adjustments`, with both the base
  and overlaid value recorded and the overlay id attached only to the
  threshold actually used to classify the event.
- **Entity adverse verdict roll-up (§5.6, `business_nested/rollup.py`).** Six
  of the twelve rules (SCOPE.md): AE-R-01 (any disqualifying event),
  AE-R-02 (**the count rule** -- >=3 recent minor events, attributes to the
  whole set), AE-R-06/07 (aggregate unsatisfied amount, material and
  disqualifying), AE-R-11 (all-immaterial floor) and AE-R-12 (the
  peripheral cap, with the write-off/fraud-marker bypass). Attribution is a
  flat, parallel `(entity_id, event_id, rule_id)` table, one row per
  contributing event -- the same shape for a single-event rule and a
  count-based one (spec 05 §13 Q6).
- **Entity scoring (§5.7, `business_nested/scoring.py`).** One scorecard
  family for both natural persons and juristic entities (SCOPE.md: "for one
  family"), built with `decider`'s `ScorecardConfig`, calibrated through
  project 00's `core.calibration` keyed by segment (`SEGMENT_SME_PEOPLE` /
  `SEGMENT_SME_ENTITY` -- both already present in 00's calibration table), and
  a project-05 risk-grade `DecisionTableConfig` (00's own risk-grade table
  doesn't cover business products/segments -- see "Gaps in what I consumed").
  An entity-level, segment-scoped overlay demonstrates the "same overlay
  mechanism, applied per entity rather than per application" property
  (§5.7's own distinguishing requirement). **Not built**: the thin-file / no-
  hit / no-enquiry-possible situation split (§5.7's four situations) -- every
  entity runs through the full scorecard.
- **The people blend (§5.8, `business_nested/blend.py`).** PP-01 (inclusion),
  PP-02 (75% coverage requirement), PP-03 (base weights, owner ownership plus
  a capped non-owning-controller notional weight), PP-04 (control weighting,
  >=60% floor with renormalisation), PP-05 (log-odds blend, not an average of
  grades), a three-item subset of PP-06 (any disqualifying entity declines
  the business regardless of blend inclusion -- PP-07; a weak-grade cap; a
  critical-entity-material floor) and PP-11 as an aggregate delta rather than
  a full per-entity-per-overlay decomposition table. **Not built**: PP-08/
  PP-09 (surety and guarantor cover) -- no security/cover model exists in
  this slice at all (structuring is a single rate lookup, SCOPE.md).
- **Business financial assessment (§5.9, `business_nested/financial.py`),
  three ratios** (SCOPE.md): interest cover, current ratio, gearing, through
  a second small `ScorecardConfig` and project 00's calibration again (a
  third "same capability, different settings" instance in this project,
  after entity scoring and the risk-grade lookup). Not built: the 48-line
  spread, sector benchmarking, the audit-level haircut table, the bank-
  statement fallback.
- **The combined business grade (§5.10, `business_nested/grade.py`).**
  Financial and people, blended on log-odds, with a **reduced overlay
  stack**: the event-threshold overlay (position 1) and one business-level
  PD-multiplier overlay (position 6, spec's own numbering) -- two of eleven,
  not all eleven (SCOPE.md).
- **Structuring and pricing (§5.12, `business_nested/pricing.py`), a single
  lookup** (SCOPE.md): one appetite-cap table, one grade x security-type rate
  table, `core.instalment`/`core.fees`/`core.rounding` (project 00,
  unmodified) for the offer. No circular amount<->rate<->security-type
  solve, no candidate search.
- **Sole proprietors (§5.2, §5.12 item 7, `business_nested/sole_proprietor.py`).**
  A real call into project 02's `pipeline.build()`, gated on
  `legal_form_code == 1`; a statutory fail is a hard decline.
- **Single-event counterfactuals (§9.4) and the partial one-entity
  re-assessment (H5, acceptance §10 item 11, `business_nested/counterfactual.py`)**,
  both explicitly required by SCOPE.md ("because 11 depends on it", spec 11
  §5.17.5). The counterfactual re-scores the whole pipeline with one event
  removed. The partial re-assessment is a genuine partial recompute, not a
  full re-run relabelled: it re-derives structure, event classification and
  scoring for **only** the refreshed entity (skipping the event-threshold
  table lookup and the scorecard/calibration/risk-grade lookup for every
  other entity, which is where this stage's cost sits), splices the result
  into the rest of the already-known entity table, and re-runs the cheap,
  non-ragged application-level stages. Proven equivalent to a full re-run in
  `tests/test_counterfactual.py`.
- **Ordering independence** (acceptance §10 item 3) is proven by shuffling in
  `tests/test_pipeline.py` and `tests/test_structure.py`/`test_rollup.py` --
  every list `structure.py` emits is sorted by `entity_id`/`(entity_id,
  event_id)` regardless of request order, so nothing downstream has to
  re-sort to stay order-independent.
- **Determinism and batch/record-at-a-time agreement** (§13 Q16) are proven
  in `tests/test_pipeline.py::test_batch_mode_agrees_with_record_at_a_time`
  and `test_replay_reproduces_the_decision_bit_for_bit` -- the same
  `frame_step`s run identically under `.score()` and `.run()`.

### What I left out

- The 14-rule entity disqualification matrix and the 20-rule business
  disqualification matrix (§5.3, §5.4's E-AROD table) -- not named in
  SCOPE.md's slice for 05.
- Full graph collapsing (cycle detection, the materiality-floor-driven
  3-level expansion) -- the caller supplies pre-resolved per-path ownership;
  see "Spec problems".
- Two scorecard families (natural person vs. juristic) and the four scoring
  situations -- SCOPE.md: "entity scoring for one family".
- PP-08/PP-09 (surety and guarantor cover), the full appetite/security/group-
  exposure stage (§5.11), and the pricing circularity/search (§5.12) --
  SCOPE.md caps structuring at "a single product 50 lookup".
- The financial spread beyond three ratios (§5.9) -- SCOPE.md's own cap.
- The committee pack (§9.3) -- explicitly skipped, SCOPE.md.
- The internal-vs-communicable reason-code split (§9.2) -- every reason code
  in this slice is treated as internal; no client-facing wording layer.
- Nine of the twelve roll-up rules, nine of eleven overlay stack positions,
  most of the qualitative-override and behavioural-component machinery at
  §5.10 -- all named explicitly as reductions in SCOPE.md.

---

## 2. Reuse

**From project 00 (`credit_core`), unmodified:** `core.adverse_events.event_severity_code`
and `.event_age_months` (called directly, per-event, inside `events.py`/`structure.py`);
`core.adjustments.AdjustmentRegister`/`Adjustment`/`AdjustmentEffect` (four
independent overlay registers: event thresholds, entity score, business PD, plus
02's own capacity overlay reached through the sole-proprietor call);
`core.calibration.build_calibration_table()`/`probability_of_default_step`
(three separate uses: entity scoring, financial scoring, and indirectly through
02); `core.instalment`, `core.fees`, `core.rounding` (pricing, unmodified);
`core.reason_codes.ReasonCodeRegistry` (this project's own registry, built the
same way 00's is); `core.evidence.cell_id` (every table this project defines).
`credit_core.adverse_events._ALWAYS_DISQUALIFYING` is reused directly (the
underscore-prefixed, non-public set) for the AE-R-12 write-off/fraud bypass --
the same "reach past the one stable entry point" trade-off 02's NOTES.md
documents for `core.obligations._process`.

**From project 02, as a whole pipeline:** `pipeline.build()` is invoked, real
record in, real record out, for every sole-proprietor application -- not a
stub, not a re-derivation of affordability logic.

**From `decider`'s built-ins:** `DecisionTableConfig` (five tables: event
thresholds, two risk-grade boundary sets, the appetite cap, the rate card);
`ScorecardConfig` (two: entity scoring, financial scoring); `frame_step`
(every stage that has to loop over a ragged collection: structure resolution,
event classification, the roll-up, entity scoring, sole-proprietor gating);
`dag`/`flow`/`step`; `param()`/`missing_as()` throughout; `.relabel()`
extensively; `Engine().bind()` used a **second** way this project's
predecessors didn't need -- bound once at import time and called from
*inside* a `frame_step`'s Python body, to run a real `DecisionTableConfig`/
`ScorecardConfig` sub-pipeline once per element of the outer ragged
collection (see "Framework friction" -- there is no first-class construct
for this).

**Written from scratch:** the whole nested-to-flat unpacking in `structure.py`
(spec 05's own dominant difficulty); the roll-up rule set (`rollup.py`); the
blend arithmetic (`blend.py`); the entity/financial scorecard *content*
(the `ScorecardConfig` mechanism is reused, the characteristics are project-
specific); the project-05 risk-grade and rate-card tables (project 00's
equivalents don't cover business products/segments); the counterfactual and
partial-re-assessment logic.

---

## 3. Gaps in what I consumed

- **00's risk-grade table doesn't cover business products/segments.**
  `credit_core.risk_grade.build_risk_grade_table()`'s `_BOUNDARIES` dict only
  has rows for `(product_code, segment_code)` in `{(10,1),(10,2),(10,3),
  (20,1),(20,2)}` -- products 50/51 and segments 4/5 (SME_PEOPLE/SME_ENTITY)
  are absent, even though `credit_core.vocab` **already declares**
  `SEGMENT_SME_PEOPLE`/`SEGMENT_SME_ENTITY` and 00's calibration table
  **already has rows for both**. I could not add rows without editing 00
  (read-only), so `scoring.py` and `grade.py` build their own small
  risk-grade `DecisionTableConfig`s, same mechanism, project-05-owned data.
  00's `core.appetite` has the identical gap (only products 10, segments 1/2)
  -- `pricing.py` builds its own appetite-cap table for the same reason.
  This is not a defect in 00 (its own NOTES.md is explicit that these tables
  are "working depth, a handful of cells" for its own slice) -- it is a real
  seam a second consumer hits immediately, and it is worth 00 knowing that
  the *vocabulary* it already declared (the SME segments) creates an
  expectation its *tables* don't meet yet.
- **`core.adverse_events`'s 14-type enum doesn't match spec 05 §4.4's own 14
  types.** 00's types (`JUDGMENT`, `DEFAULT_LISTING`, ..., `REGULATORY_FINDING`)
  are a different list from spec 05's own ("civil judgment", "dishonoured
  payment", "rental default listing", ...). Since the classification
  *mechanism* (satisfied / amount / status-only / always-disqualifying) is
  identical regardless of the specific 14 labels, I reused 00's enum directly
  as project 05's own event-type vocabulary rather than inventing a second,
  parallel numbering for the same four behaviours -- see "Spec problems" for
  why I think this is a spec inconsistency rather than a real design choice
  05 was supposed to make twice.
- **Project 02 has no uniquely-named importable entry point** -- see
  "Framework friction" below; not a defect in 02, but a real consumption gap
  every downstream project (05, then 11 consuming 05) will hit identically.

---

## 4. Framework friction

Four items confirmed with a minimal reproduction; two required reading
`decider` internals directly (BRIEF: "if you had to read internals, say so").

### 4.1 A step's own typo guard can refuse a completely unrelated, legitimate new input column

`decider/registry/resolve.py` uses `difflib.get_close_matches(name, candidates,
cutoff=TYPO_CUTOFF)` (`TYPO_CUTOFF = 0.8`) against every name **already
produced earlier in the same scope** when a step reads a name nothing has
written yet. If the new name is >80% similar (by `difflib.SequenceMatcher`
ratio) to an earlier output, `decider` assumes it is a typo and refuses to
bind it as a new input column at all -- not a warning, a hard `WiringError`:

```
WiringError: t/entity_scorecard/entity_bureau_score: input 'entity_bureau_score'
is not produced by any earlier step and is not a declared input column. Did you
mean 'entity_base_score' (produced by 't/entity_scorecard/entity_base_score')?
Rename it, or relabel(reads={'entity_bureau_score': 'entity_base_score'}).
```

`entity_scorecard.py`'s own constant offset (`entity_base_score`, a `600`
literal with no relation at all to `entity_bureau_score`, a genuine new input)
happened to have a 0.833 similarity ratio -- just over the 0.8 cutoff. I
confirmed this with a bisected minimal reproduction (six variants, changing
one name at a time): the *only* thing that mattered was the string
similarity between the two names, not their meaning, their declared type, or
which one was genuinely new. Renaming `entity_base_score` to
`entity_scorecard_offset` (ratio 0.57) fixed it with no other change.
This is a real, sharp-edged trap for exactly the domain this project is
in: every scorecard in this spec set has a constant "base"/"offset" term and
a `bureau_score`-shaped characteristic, and `base`/`bureau` are close enough
in English that this will recur. Worth either raising `TYPO_CUTOFF`,
excluding constant-node outputs from the candidate set, or at minimum
documenting the threshold next to `ScorecardConfig`'s own docstring.

### 4.2 A `list[date]` column silently degrades to raw integers between two `frame_step`s

00's and 02's NOTES.md both document that a `list[struct]` **output** from a
`frame_step` crashes result materialisation. This project found a third,
distinct instance of the same underlying gap, for a completely different
type: a `list[date]` column **written by one `frame_step` and read by a
second `frame_step` downstream in the same `dag`** silently degrades to
`Array(Int64, ...)` (raw epoch-day integers) at the boundary between them --
no error, no warning, until something calls `.year` on a plain `int`:

```python
@frame_step(reads=["entities"], writes=["ev_date"])
def stage1(df):
    return df.with_columns(pl.DataFrame([{"ev_date": [ev["d"] for ev in row["entities"]]}
                                          for row in df.select("entities").to_dicts()]))

@frame_step(reads=["ev_date"], writes=["ev_year"])
def stage2(df):
    # df.schema here: {'ev_date': Array(Int64, shape=(2,))} -- not List(Date) any more
    ...  # AttributeError: 'int' object has no attribute 'year'
```

Constructing the intermediate `pl.DataFrame(results)` directly (outside
`decider`) types the column correctly as `List(Date)` and round-trips fine
through `.to_dicts()` -- the corruption happens specifically inside
`decider`'s own state materialisation between the two steps. Workaround
(`structure.py`): derive any date arithmetic (`event_age_months`) **inside**
the same `frame_step` invocation that still holds real `datetime.date`
Python objects, and never emit a `list[date]` column for a later step to
read. This is a materially different, and more dangerous, failure mode than
the `list[struct]` crash: that one fails loudly at `.run()`/materialisation
time; this one fails silently at `.score()` time too, and only surfaces
downstream, in whatever code happens to call a date method on what looks
like a normal list.

### 4.3 An `Adjustment`'s `kind` doesn't have an obviously-correct choice for a materiality threshold

00's `_TIGHTEN_RULES` (`credit_core/adjustments.py`) recognises
`buffer_adjustment` as tightening only via `op="add", value>=0` (a buffer
that only grows) and `cap_adjustment` as tightening via `op="multiply",
value<=1.0` (a ceiling that only shrinks). Spec 05's own worked example
("halve the judgment materiality threshold") is a ceiling shrinking --
`cap_adjustment`, not `buffer_adjustment`, even though "a threshold you
compare an event amount against" reads far more naturally as a buffer than
a cap on first encounter. My first attempt (`kind="buffer_adjustment"`)
raised at construction time (`ValueError: ADJ-05-014: buffer_adjustment is
tighten-only but its effect loosens`) -- a clear, actionable error, but one
that requires knowing 00's *other* five kind names and their directions to
resolve, since the message doesn't suggest the alternative. A cross-
reference table in `adjustments.py`'s own docstring (which `kind` fits
which shape of "the good direction") would have saved the detour.

### 4.4 No construct runs a step, or a sub-pipeline, once per element of a collection nested inside a row

Spec 05 §13 Q1 asks this directly: "what is the unit that operates on one
element of a collection, when the element itself contains a collection?"
`decider` has `frame_step` for "arbitrary Python over a whole application
row, including its ragged fields" -- and nothing narrower. Reusing a real
`DecisionTableConfig`/`ScorecardConfig` **per entity** (not per application)
required binding a second, small `Engine` at import time and calling
`.run()` on it from *inside* the outer `frame_step`'s Python body
(`events.py`'s threshold lookup, `scoring.py`'s entity scorecard). This
works, in both `.score()` and `.run()` (confirmed empirically before
committing to the pattern, including the zero-entity edge case), and it is
genuine reuse of the real engine and the real config objects -- but it is a
workaround assembled from two `Engine`s wired together by hand, not
something the framework offers as a documented composition primitive. The
answer this project landed on for spec 05 §13 Q1: **a `frame_step` at each
nesting level, with a nested `Engine` call standing in for "run this once
per element"** -- and every attribution downstream is kept in flat,
relational (long-form) tables joined by id, because (§4.5 below) no step can
*emit* a nested column anyway, so the flat representation is forced
regardless of how the per-element work gets done.

### 4.5 (confirms 00/02, extends it) `list[list[T]]` outputs crash exactly like `list[struct]` outputs

00's NOTES.md documents that a terminal `frame_step` output typed
`list[struct]` crashes materialisation. A `list[list[int]]` output --
structurally simpler, no struct involved -- crashes the same way, with a
lower-level error:

```
thread '<unnamed>' panicked at crates/polars-core/src/fmt.rs:1282:9:
nested Objects are not allowed
```

This confirms the constraint is genuinely "no nested list output from any
step", not specifically "no struct". It is the reason every attribution
table in this project (`rollup.py`'s `attr_entity_id`/`attr_event_id`/
`attr_rule_id`, `events.py`'s per-event arrays) is a set of flat, equal-
length parallel lists rather than one list-of-lists-per-entity column.

### Smaller things

- **Every project's entry point shares the filename `pipeline.py`
  (BRIEF's own convention).** Consuming *another* project's `build()`
  function (not its library package, which is the 00-style pattern) by a
  plain `import pipeline` risks silently resolving to the wrong project's
  file depending on `sys.path` order -- and the risk is not hypothetical: it
  applies equally to a module reaching for *its own* project's
  `pipeline.py` from inside its own package (`counterfactual.py` needed
  `pipeline.build()`; a plain `import pipeline` there is exactly as fragile
  as reaching into project 02's). Both `sole_proprietor.py` and
  `counterfactual.py` load their target `pipeline.py` via
  `importlib.util.spec_from_file_location`, by an explicit path (found on
  `sys.path` for 02's, resolved relative to `__file__` for this project's
  own) rather than by module name. Project 11, consuming this project the
  same way project 05 consumes 02, will need the identical workaround.
- **`step()` with multiple outputs needs a plain tuple with an explicit
  `-> tuple[...]` return annotation** (not a dict, even though every output
  has a name declared right there in `outputs=(...)`). `blend.py` and
  `outcome.py` both keep a dict-returning "real" function for readability
  and testability, with a thin positional-tuple adapter wrapping it for
  `step()` -- doubling the function count for every multi-output step. A
  `step()` variant that accepted a dict return keyed by the declared
  `outputs` names (or inferred them) would remove that whole pattern.
- **A `list`-typed input to a plain `step()` arrives as a `numpy.ndarray`
  in its row loop**, confirmed again here (00 NOTES.md 4.4) for a method
  the earlier finding didn't cover: `.index()` doesn't exist on an
  `ndarray`, only `is None`/truthiness were flagged before. `list(...)` at
  the top of the function is the fix every time; worth a blanket note in
  `missing_as`'s docstring that *any* list method beyond iteration needs
  this, not only truthiness checks.
- `DecisionTableConfig`'s `default: [...]` row (used here for the event-
  threshold table's fallback for event types outside the declared five) is
  exactly the right mechanism and worked first try, once the ±inf-bound
  convention (00's own finding) was already known from reading 00's code.

### What worked well

The nested-`Engine`-inside-a-`frame_step` pattern, once found, composed
cleanly and was fast to iterate on: real `DecisionTableConfig`s and
`ScorecardConfig`s, genuinely reused, with no special-casing for "this one
runs per entity instead of per application". `AdjustmentRegister.apply_stack`
(the non-`_step` form) made the entity-scoped and event-scoped overlays
(different provenance keys from `apply_stack_step`'s built-in four) trivial
to wire without forking the mechanism. `.emit()`'s "input columns plus
whatever nothing reads passes through automatically" meant `decision_id`
needed zero wiring to survive end to end. `pipeline.build().parameters().defaults()`
produced a correct, complete `params.json` in one call, exactly as 00/02's
NOTES.md describe.

---

## 5. Spec problems

- **Spec 05's own 14 event types (§4.4) don't line up with project 00's
  `core.adverse_events` 14-type enum**, even though 05 is supposed to
  *consume* that capability (DEPS.md: "`core.adverse_events`... for sole
  proprietors, `core.affordability`"). 00 was built first (wave 0) and
  necessarily invented its own illustrative 14 types without knowing spec
  05's list in advance -- but the spec set doesn't flag this collision
  anywhere, and a careful implementer following DEPS.md literally ("consume
  `core.adverse_events`") has to choose between forking the enum (two
  parallel 14-type lists for the same four classification behaviours) or
  reusing 00's as I did. Worth a line in 00-ADDENDUM or spec 05 itself
  saying which one is authoritative.
- **SCOPE.md's "structure resolution (§5.1, a natural person reached by two
  paths is evaluated once)" reads as asking for the whole §5.1 stage**,
  including the graph-to-bounded-structure collapse, but the *line budget*
  (1652 lines for the whole slice, alongside six other full stages) doesn't
  fit a real cycle-aware graph traversal plus everything else SCOPE.md
  explicitly asks for. I read the parenthetical as SCOPE.md's own
  clarification of *which part* of §5.1 is load-bearing for this slice (the
  de-duplication behaviour) rather than the whole stage, and left the
  graph-collapse itself as a documented gap -- a different implementer
  reading the same sentence could reasonably have spent the whole budget on
  the graph traversal alone and built a thinner roll-up.
- **§5.4's criticality table cites a "corporate guarantor providing more
  than 20% of required cover" clause for the Critical class**, which depends
  on a cover/security model (§5.11, PP-08/PP-09) that SCOPE.md explicitly
  puts out of scope for this slice ("take structuring... only as far as a
  single product 50 lookup"). The criticality table as SCOPE.md scopes it is
  therefore internally under-specified by one clause with no way to satisfy
  it inside the slice's own boundary -- I dropped the clause and noted it,
  rather than building a cover model solely to serve one criticality rule.
- **AE-R-08 ("recent_adverse... blocks entity grades 1-3") isn't in the
  six-rule subset SCOPE.md asks for**, but PP-06's worst-of table (§5.8)
  references `recent_adverse` directly as one of its five conditions. I
  substituted "any critical entity at material-or-worse" for the missing
  `recent_adverse` flag in `blend.py`'s PP-06 subset, which is close in
  spirit but not the same test (recency vs. severity) -- a real, documented
  approximation rather than a faithful implementation of the cross-reference
  SCOPE.md's own reduction leaves dangling.

---

## 6. What I would do next

1. Extend the six-rule roll-up subset to the full twelve, and the entity
   scoring split to both scorecard families -- both are purely additive over
   the shapes already built (the roll-up rule dispatch and the nested-
   `Engine` scoring pattern don't change).
2. Build the actual graph-collapse (cycle detection, the 3-level expansion
   with the 5% materiality floor) in `structure.py`, replacing the
   caller-supplied-`effective_ownership_pct` simplification -- the highest-
   value gap, since it's the one piece of §5.1 genuinely not built.
3. A security/cover model (§5.11, PP-08/PP-09) so the criticality table's
   corporate-guarantor clause and the pricing stage's cover-ratio-driven
   security type can both be built for real, rather than the flat
   `security_type_code` input this slice takes as given.
4. The internal-vs-communicable reason split (§9.2) -- every field the
   communicable rendering needs (which codes are third-party, the
   attribution triple) is already in this project's output; only the
   registry's communicability flag and the second rendering pass are
   missing.
5. Push 4.1 (the typo-guard false positive) and 4.2 (the `list[date]`
   silent degradation) upstream -- 4.2 in particular is a correctness trap
   with no error message at the point it happens, and every one of this
   spec set's twelve projects that computes an age or a duration across two
   `frame_step`s will hit it the same way I did.
