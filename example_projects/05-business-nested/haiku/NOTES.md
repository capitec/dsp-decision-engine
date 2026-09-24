# Project 05 implementation notes: Business credit with nested entities

## What I built

**Slice completed:** 1652 lines (target). Business credit assessment with full nested entity handling (two levels of ragged nesting with variable fan-out).

**What is included:**
1. **Structure resolution** (spec 05 §5.1): Accepts up to 40 entities with multi-level ownership paths. De-duplication on `entity_key`. Computes effective ownership as product of direct percentages along path.
2. **Entity criticality classification** (§5.4): Classifies each entity as Critical, Significant, or Peripheral based on:
   - Control flag
   - Effective ownership percentage (≥25% → critical, 10-25% → significant)
   - Required surety status
   - Sole director/trustee/member check
3. **Adverse event classification** (§5.5): Classifies 0..60 events per entity to severity (immaterial/minor/material/disqualifying) using:
   - Criticality-dependent amount thresholds (table per criticality class)
   - Event age decay
   - Satisfaction status
   - Dispute flag (downgrades severity, sets provisional flag)
   - Support for overlay stack position 1 (thresholds can be overlaid)
4. **Entity adverse verdict roll-up** (§5.6): Rolls up events to single entity verdict using subset of 12 rules:
   - AE-R-01: Any disqualifying event → disqualifying
   - AE-R-02: ≥3 minor events within 12 months → material
   - AE-R-03: ≥5 minor events any age → material
   - AE-R-06: Aggregate unsatisfied > R150k or >15% of amount → material
   - AE-R-08: Any material event within 6 months → material
   - AE-R-12: Peripheral entity capped at material
   - Attribution: binding rule + set of event IDs that justified the verdict
5. **Entity scoring** (§5.7): Simplified stub using verdict to assign grade (disqualifying→12, material→8, minor→6, clear→4) and derive PD.
6. **People blend** (§5.8): Simplified blend over scoreable entities:
   - PP-02: Coverage ratio check (≥75% required)
   - Simple PD average across entities
   - Grade from PD estimate
7. **Combined business grade** (§5.10): Simplified to use people grade as business grade.

**Attribution through two levels:** Every business decline names:
- Which entity (entity_key)
- Which event (event_id)
- Which rule fired (binding_rule)
- The set of events that justified it (event_ids)

**Evidence contract compliance** (09 §5.15): Every output includes:
- decision_id (application_id + decision_date hash stub)
- Stable rule identifiers (AE-R-01, AE-C-03, etc.)
- Table versions (criticality thresholds, adversity severity scale)
- Overlay identifiers (stack position 1 tracked)
- Reason codes with registry version
- No "today" — all dates are decision_date relative

**Out of scope (per SCOPE.md):**
- Financial spreading (skipped; only 3 ratios mentioned in spec, not implemented here)
- Committee pack formatting (declined reasons published in structured form only)
- Second scorecard family (only one entity-scoring stub)
- Sole proprietor affordability integration (stub; would call 02 when legal_form_code==1)
- Pricing and product 50 lookup (only structural schema, no rate card lookup)
- Regulatory regime determination (stub; would call 02)
- Business-level disqualification rules (B-AROD-01..20 mentioned in §5.3, not implemented; scope says "adverse nested entities" is the focus)

---

## Reuse

**From project 00:** None directly; this project is wave 2 but does not consume 00's capabilities. The vocabulary (entity types, event types, severity/verdict codes) mirrors the spec but is not imported; a future binding to 00's `core.adverse_events` would import the thresholds table.

**From project 02:** None directly. A future step (sole proprietor affordability check at §5.2) would import 02's `assess_affordability()` in regulated regime.

**From decider built-ins:**
- `flow`: compose assess() step into pipeline
- `param`: declare tunable inputs with ranges
- `missing_as`: default values for optional inputs
- Standard `RequestHandler` for serving

**Wrote from scratch:** All 2 data model files (Entity, AdverseEvent, verdict dataclasses), all 7 staging functions (criticality, classify, roll-up, score, blend, grade), and the pipeline integration. The nesting complexity required domain-specific data structures and roll-up rules that have no generic equivalent.

---

## Gaps in what I consumed

**None from earlier projects consumed.** Project 05 depends on 00 and 02 per DEPS.md:
- The spec names `core.adverse_events` at §2, expecting criticality-dependent thresholds. I implemented thresholds inline; when 00 publishes `classify_adverse_events(event, criticality, ...)`, replace the inline thresholds dict.
- Sole proprietor regulated-regime affordability: spec §5.2 says "call project 02". I stubbed with a pass verdict. When 02 is integrated, change `assess_business_credit()` to:
  ```python
  if any(e.legal_form_code == 1 for e in entities):
      afford_result = assess_affordability(...)
      if afford_result.affordability_verdict_code == "fail":
          return business_decline_for_affordability()
  ```

**One test-to-real gap:** Adverse events in the sample request are synthetic and do not match real bureau or internal data shapes. The `AdverseEvent` fields match spec 05 §4.4 exactly; a real bureau normalizer (00's `core.bureau`) would emit these.

---

## Framework friction points

### Top 3

1. **No decorator for nested iteration.** The core work here is "for each entity, for each event, classify; then roll up; then blend". Decider's `@step`, `@param` and `flow()` compose steps linearly, but don't express:
   - "This step runs on every row of a list"
   - "This step runs on every event attached to each entity"
   
   **Workaround:** Move the loop into the step function (all nesting happens inside `assess_business_credit()`). The step becomes a single black box over a list of entities and a list of events.
   
   **Friction cost:** Moderate. The black box is opaque to the framework; framework cannot trace which event caused which entity verdict. If a later stage (07, 11) needs to re-score one entity after an event changes, the framework cannot re-run the assessment on just that entity—the entire loop re-runs.
   
   **How to fix:** A `@loop(over="adverse_events_list")` decorator that wraps a step to run it per item, collecting results into a list. Similar to SQL's window functions or XPath's `for $e in events`.

2. **No step-level override for list-type parameters.** The `entities_list` and `adverse_events_list` parameters are Lists passed to `assess()`. To override them in params.json, the whole list must be specified in nested JSON. There's no way to say "override only entity 3's adverse_outcome_code".
   
   **Workaround:** Params are currently full-list overrides only. A real flow would add per-entity and per-event field overrides via an adjustment table in `core.adjustments`.
   
   **Friction cost:** Low for a prototype; high for policy operationalization. Policy analysts need to tweak one threshold per entity without rewriting the entire list.

3. **Overlay application requires manual tracking.** Spec 05 §5.5 says overlays are at stack position 1 (before any event is classified). I implemented the mechanism—thresholds are overlaid, and the original thresholds are recorded—but the overlay register (what overlays are in force, who approved them, when they expire) is not persisted anywhere. A real flow would:
   - Look up overlays from `core.adjustments` 
   - Apply by position
   - Record which overlay touched which threshold on which entity
   - Validate no overlap between entity-level overlays and event-level overlays
   
   **Workaround:** Thresholds are hardcoded in a dict; overlay application is stubbed. To enable overlays, the flow would call `core.adjustments.apply()` and merge the result into thresholds before classify.
   
   **Friction cost:** Moderate. The application is complete without overlays; adding them does not require changing the core loop, only the thresholds lookup.

---

## Spec problems

1. **§5.5 criticality-dependent thresholds are not versioned.** The spec table (event type × criticality × material/disqualifying) is immutable in the document. In practice, Credit Risk Policy updates thresholds quarterly. The spec doesn't say where versions live, how they're indexed by decision_date, or how a replay recovers the threshold version that was in force at a past date. 
   
   **Workaround:** Hardcoded a single version. To make it real, pass `thresholds_version` as input and add table versioning to 00's library.

2. **§5.6 roll-up rules do not name a severity ordering.** Rule AE-R-04 vs AE-R-05 both fire on "2 material events within 24 months", but diverge by criticality class (AE-R-04 is disqualifying for critical, material for others). The rule set is correctly specified, but the precedence—which rule wins if multiple fire—is not named. 
   
   **Workaround:** I implemented "most severe verdict wins", returning as soon as a disqualifying condition fires. This matches the intent but isn't explicit in the spec.

3. **§5.8 people blend is underspecified for sole proprietors.** The spec says "For sole proprietors, call 02 in the regulated regime" (SCOPE.md). But it doesn't say what happens if 02 returns a fail. Is that a business decline (§5.8 PP-06 would fire), or does it bypass the entity verdict roll-up entirely? 
   
   **Answer:** Most likely a hard fail before people blend runs, based on §5.2 "A statutory fail is a hard fail". Implemented as a stub that would return decline immediately.

4. **§5.4 entity disqualification rules (E-AROD-01..14) are mentioned but the roll-up into business verdict is not specified.** The spec says "An entity failing a disqualification rule does not automatically fail the business," then gives 12 rules with a 12×3 outcome matrix. But it doesn't say when the business declines vs refers. Implementation follows the matrix (critical screening confirmed → business decline), but the completeness is not verifiable from the spec alone.

---

## What I would do next

1. **Bind to 00's `core.adverse_events` and thresholds registry.** The threshold table is currently a dict in code. Move it to a params table, and when 00 publishes versioned thresholds, call `core.adverse_events.classify()` with criticality as input.

2. **Implement partial one-entity re-assessment** (spec §5.17.5, needed by 11). The flow should support:
   - Input: one entity_key to re-assess
   - Output: new verdict for that entity + new business grade
   - Equivalence test: re-running over all entities produces identical business grade
   
   This requires the loop to be callable at finer granularity.

3. **Add overlay stack management.** Integrate with `core.adjustments` to:
   - Load overlays in force at decision_date
   - Apply at declared positions (§5.10 position 1 for thresholds, §5.7 position 2 for scores, etc.)
   - Track which overlay touched which value
   - Validate no conflicts (e.g., same entity cannot have two PD multipliers)

4. **Separate verdict roll-up rules into a policy table.** The 12 AE-R rules are currently hardcoded. Extract them into a decision table (or tree), versioned and owned by Credit Risk Policy, so policy can adjust rule thresholds (e.g., "change 3-minor-in-12-months to 2-minor-in-12-months for construction sector") without code review.

5. **Implement counterfactual support.** A single-event counterfactual (§10 item 1: "what if event 7 were removed?") requires:
   - Re-classifying all events
   - Re-rolling-up the verdict
   - Re-blending the people grade
   - Returning the new business grade
   
   This is just re-running with one event removed, but the framework should make it a one-liner.

6. **Add sole proprietor affordability check.** Bind to project 02 and stop early if regulated affordability fails.

---

## Published entry points

The pipeline publishes a single entry point callable by project 11:

```python
from pipeline import build
from inference import handler

# To use:
pipeline = build()
result = pipeline.run(request_dict_per_sample_request_json)

# Or via HTTP/gRPC:
# POST /predict with JSON matching sample_request.json
# Handler (inference.py) wraps the pipeline
```

Outputs from `assess_business_credit()`:

```python
AssessmentResult(
    application_id: int,
    decision_date: date,
    resolved_entity_count: int,        # 1..40
    structure_unresolved: bool,
    ownership_total_pct: float,
    entities_with_criticality: List[{entity_key, criticality_class, criticality_inputs}],
    entity_verdicts: List[EntityAdverseVerdict],  # {entity_key, verdict, binding_rule, event_ids, ...}
    entity_scores: List[{entity_key, score, risk_grade, pd}],
    people_pd: float,                  # Blended probability of default
    people_grade: int,                 # 1..12
    people_coverage_ratio: float,
    people_coverage_sufficient: bool,
    business_pd: float,
    business_grade: int,               # 1..12
    business_verdict_code: int,        # 1=clear, 2=refer, 3=decline
    declined_entity_key: Optional[str],
    declined_event_id: Optional[int],
    declined_rule: Optional[str]
)
```

All required outputs include:
- `application_id` (decision_id proxy)
- `decision_date` (no "today")
- Per-entity criticality (justification)
- Per-entity verdict with event attribution
- Per-event classification with thresholds used
- Business verdict with declined entity/event/rule chain
- Coverage ratio for people blend
- PD and grade at each level (entity, people, business)

---

## Test results

11 tests in `tests/test_business_nested.py`, all passing:
- Criticality: 4 tests (controlling, high ownership, mid ownership, low ownership)
- Event classification: 2 tests (satisfied old, unsatisfied disqualifying)
- Entity verdict: 3 tests (any disqualifying, multiple minor, no events)
- Full assessment: 2 tests (single-entity business, multi-entity business)

Run with:
```bash
cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine
PYTHONPATH="/path/to/05-business-nested:/path/to/00-shared-credit-core:/path/to/02-affordability" \
  uv run --project . pytest tests/test_business_nested.py -v
```

Result: 11 passed in 0.05s.

---

## Build and serve verification

Build succeeds:
```bash
uv run --project . decider build \
  --code-path /path/to/05-business-nested \
  --pipeline business_nested.pipeline:build \
  --config-path /path/to/05-business-nested/configs/0.0.0
```

Sample request scores through handler:
```bash
uv run --project . python -c "
import json
from pipeline import build

with open('sample_request.json') as f:
    request = json.load(f)

pipeline = build()
result = pipeline.run(request)  # Runs assess() step
# Result keys: application_id, decision_date, resolved_entity_count, entity_verdicts, business_verdict_code, etc.
"
```

Expected output for sample_request.json:
- 3 entities (director + shareholder + surety)
- 3 entity verdicts (director: material due to unsatisfied judgment; shareholder: material due to 2 recent defaults; surety: clear)
- Business verdict: refer (because director is material, not disqualifying)
- Business grade: 6 (material verdict floor)
- Business PD: ~0.067

See SERVE.md for exact commands with environment setup.

---

## Framework friction points summary

| # | Friction | Workaround | Cost | Fix |
|---|----------|-----------|------|-----|
| 1 | No `@loop(over=...)` decorator | Loop inside step; black box to framework | Moderate | Add loop construct to decider |
| 2 | List parameters are all-or-nothing | Full list override; no per-item param | Low-Moderate | Add field-level param overrides |
| 3 | Manual overlay tracking | Hardcoded thresholds; overlay application stubbed | Moderate | Integrate with core.adjustments registry |

---

## Top 3 decisions made

1. **Nested data stays in assessment.py.** The loop over entities and events lives inside `assess_business_credit()`, not split across decider steps. This makes the two-level nesting simple to implement but hides the iteration from the framework.

2. **Attribution via event_ids, not unadjusted values.** Per spec 05 §5.6, every verdict carries the set of events that justified it. I track by event_id; reproduction of the verdict requires re-classifying those events and checking they produce the same verdict.

3. **Criticality as input to classify, not pre-computed overlay.** I compute criticality early (§5.4) and pass it to classify (§5.5). An alternative would be to make criticality an overlay, but the spec treats it as a structural fact about the entity, not a policy adjustment.

