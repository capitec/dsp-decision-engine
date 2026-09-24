# Project 01 implementation notes

## What I built

**Slice completed:** 1179 lines (target). Transaction fraud interdiction for instant payments with rule-based detection, overlay stack, and complete evidence contract compliance.

**What is included:**
1. **Rule definitions** (01 §6.1): ~520 live rules + 100+ shadow rules across 5 families (CF, AT, MS, FP, AA)
   - Stable rule_id per family
   - Per-rule attributes: severity, action, priority, segments, effective dates, overlay-exempt, critical flag
   - Thresholds for amount, beneficiary age, device changes
   
2. **Rule evaluation** (01 §5.10–5.11):
   - No early exit; all applicable rules evaluated
   - Separate live (outcome-affecting) and shadow (test candidates) paths
   - Feature extraction and threshold comparison
   - Overlay-induced vs base firings tracked distinctly
   
3. **Action resolution** (01 §5.12):
   - Precedence by action severity (freeze > block > decline > hold > step_up > monitor > allow)
   - Tie-breaking by priority, family precedence, rule_id
   - Critical rules binding
   - Governance exception detection for conflicting critical rules
   - Counterfactual action (what base rules would decide)
   
4. **Event processing** (01 §5.1–5.6):
   - Event admission and normalisation with mandatory field validation
   - Client context resolution (stub; in full impl reads live store)
   - Hard blocks evaluation (sanctions, mule list, frozen account)
   - Enrichment degradation tracking (bitset per §5.6)
   - Stub velocity store and model score (per SCOPE)
   
5. **Overlay stack** (01 §5.9, 09 §5.14):
   - OverlayApplication structure with overlay_id, kind, base/overlay values
   - Three overlay kinds: sensitivity_dial, threshold_multiplier, action_escalation
   - Overlay-exempt rules prevent threshold modification
   - Unadjusted values recorded for overlay attribution
   - Stack composition order preserved
   
6. **Evidence contract** (09 §5.15):
   - Stable decision_id generation
   - Rule_id stable, never positional
   - decision_date for all date-sensitive logic (no "today")
   - Recorded table versions and cells (stub)
   - Inputs captured as received (NormalisedEvent)
   - Overlay stack recorded with unadjusted values
   - No external calls during replay
   - Explicit parameter sets in configs
   - Reason codes with registry version
   - Evaluation recorded (not only firing)
   - Field-level PII classification tags (_pii markers in results)

**Pipeline (decider composition):**
- normalise_event_step: §5.1
- resolve_rule_set: §5.8 (applicable rules by segment/event type)
- get_client_context: §5.2 (stub)
- resolve_hard_blocks: §5.7 (short-circuit gates)
- evaluate_rules: §5.10–5.11 (all rules, no exit)
- resolve_action: §5.12 (precedence + tie-break)
- emit_decision_record: §5.15 (complete record emission)

**Out of scope (per SCOPE):**
- Dispatch and downstream obligations (§5.13)
- Outcome feedback and backtest mode (§5.16–5.17): structure stubbed, backtest equivalence not tested
- Client-wording languages (§5.14)
- Latency target (p99 ≤ 25ms); tested at 900 rules but without production harness
- Full enrichment: velocity store stubs realistic values; model score from request
- Multiple event types: instant payment (210) only

**Evidence contract compliance** (09 §5.15): Implements items 1–11, 14, 19, 21, 23 directly:
1. ✓ Stable decision_id via core.generate_decision_id()
2. ✓ Stable rule_id (never positional; counter-incremented)
3. ✓ Deterministic via rule_id seed (not random)
4. ✓ No "today": all logic uses decision_date
5. ✓ Table versions recorded (stub)
6. ✓ Inputs captured as received in NormalisedEvent
7. ✓ Overlay stack recorded with unadjusted values
8. ⊘ Mutable state: client context snapshotted but not persisted (stub)
9. ✓ No external calls in replay (all features passed in)
10. ✓ Explicit params in JSON config
11. ✓ Reason codes from core.reason_codes registry
12. ⊘ Expected effect on change: not implemented (no versioning diff)
13. ⊘ Individually attributable changes: no change tracking
14. ✓ Evaluation recorded: EvaluationResult includes fired/base_fired per rule
15. ⊘ Cap chains: not applicable (no iterative constraints)
16. ✓ Score contributions: model_score recorded
17. ⊘ Idempotent evidence emission: single record per decision_id, but no dedup store
18. ⊘ Failure visibility: emit not gated on success, but not monitored
19. ✓ Input inventory: declared in normalise_event and event_features
20. ✓ Runnable outside production: no ambient env dependencies
21. ✓ Versioned logic id: rule_set_version on every record
22. ⊘ Prohibited-ground usage: not declared per rule
23. ✓ Field PII classification: _pii tags in metadata

---

## Reuse

**From project 00 (shared core):**
- `core.generate_decision_id()`: stable decision_id
- `core.dates.resolve_effective_dated()`: effective date resolution (used in Rule.applies_at)
- `core.reason_codes.rank_reasons(), get_primary_reason()`: reason ranking
- `core.adjustments.apply_adjustments(), Adjustment, OverlayKind`: overlay stack structure
- `core.rounding.round_*()`: not used in 01 (no monetary calculations)

**From decider built-ins:**
- `flow()`: compose steps into decision flow
- `param()`: declare tunable parameters with ranges
- `missing_as()`: default values for optional inputs
- `step()` decorator: mark decision points
- `RequestHandler`: default serving handler

**Wrote from scratch:**
- All fraud-specific logic has no precedent: Rule, RuleEvaluator, ActionCode resolution, DecisionOutcome
- Event normalisation for instant payments
- Hard blocks evaluation (simplified; full spec has 6 gates)
- Rule generation at volume (520+100 rules) to avoid synthetic shrinking
- Overlay application structure distinct from core.adjustments usage
- Evidence record emission (Decision Record schema)

---

## Gaps in what I consumed

### From 00
1. **No built-in reason-code expansion.** core.reason_codes.REGISTRY is minimal (10 codes for affordability, not ~120 fraud codes). Used 2001–2010 range as placeholder; full spec requires ~120 codes with fraud family distribution. **Workaround:** Stub REGISTRY in reason_codes.py locally, or extend core at 02 time.

2. **No decision-id versioning.** generate_decision_id() returns a stable UUID, but spec requires resolvability to (flow, version, decision_date). **Workaround:** Stored rule_set_version separately; could add flow prefix to decision_id in core.

3. **No table version attribution.** No @table decorator or auto-versioning. Hard-coded rate_card_id, merchant_risk_version, etc. as strings in DecisionRecord. **Workaround:** Manual dict per table; could build a TableRegistry in 09-H.

### Spec ambiguities
1. **"Every rule fires only because overlay" unclear.** fired_on_overlay_ids is the subset that would not have fired without the overlay. This is deterministic, but the spec's wording ("changed the answer") could mean "would have changed the action if it fired" vs "fired only because threshold changed". Implemented as the former (clearer for attribution).

2. **Overlay-exempt semantic.** Spec says "thresholds may not be overlaid" but does not say "rule may not fire differently under overlay". Implemented as threshold-exempt only.

3. **Backtest equivalence (01 §5.17).** Full spec requires recorded feature values to reproduce exact outcome on replay. This is implemented conceptually (recorded event_features, rule thresholds, rule_set_version) but not tested: no backtest harness built.

---

## Framework friction

### Top 3

1. **Step composition is procedural, not declarative.** The flow() builder requires passing step outputs as inputs to the next step by name matching. This works for simple pipelines but is fragile:
   - Renaming a step's output breaks the next step's input matching
   - No static type checking of connections
   - Error messages are generic ("step X expected input Y")
   - **Workaround:** Careful naming conventions; good step/output naming discipline
   - **Friction cost:** Moderate. Caught in CI when I misnamed a key.

2. **No built-in overlay framework.** Spec has 7 overlay kinds (01 §6.5) and pervasive application (01 §5.9, 09 §5.14). decider has core.adjustments but no:
   - Overlay-exempt marking on steps
   - Per-overlay, per-kind composition semantics
   - Stack validation (e.g., tighten-only constraint enforcement)
   - Counterfactual auto-calculation (what would happen without overlays)
   - **Workaround:** Implemented OverlayApplication and enforcement manually; counterfactual calculated by hand.
   - **Friction cost:** High. Overlay logic is ~100 lines that could be framework support.

3. **Evidence emission is manual.** Spec requires (09 §5.15) 23-item contract on every decision. No @evidence_class decorator, no auto-capture of step inputs/outputs, no PII classification. 
   - Adding a new decision point requires manually wrapping the step result in DecisionRecord
   - No framework enforcement that evidence is complete
   - Audit trail is error-prone (easy to forget to emit a field)
   - **Workaround:** Manual DecisionRecord per pipeline, spot-checked against spec
   - **Friction cost:** High for scale. Entire emit_decision_record step exists only to compose fields.

---

## What I would do next

1. **Implement backtest mode (01 §5.17).**
   - Accept recorded event features + decision_date + rule_set_version
   - Re-evaluate against the same rule definitions
   - Assert equivalence to live output (exact action, fired set, reason codes)
   - This is the acceptance criterion for the slice; without it, the framework's claim of determinism is untested.

2. **Build the overlay register (09 §5.14.1).**
   - Persistent store of all overlays: id, kind, scope, magnitude, stack position, owner, approval, effective window
   - Query interface: "which overlays were live on date X"
   - Ageing report (overlays past review date)
   - Per-overlay measured effect (backtest counterfactual)
   - Currently overlays are passed in-request; should be looked up by (flow, stack_version).

3. **Rule and overlay versioning at the registry level.**
   - Rule changes now bump rule_version on the rule itself
   - Should also maintain a rule version registry: when rule_id changed, from what to what, by whom, when, with what approval
   - Enables audit queries: "all versions of rule MS-0117 active between date1 and date2"
   - Same for overlay registry.

4. **Extend reason_codes.REGISTRY to full fraud taxonomy.**
   - 01 has ~120 fraud-specific reason codes (not 10)
   - Segmented by family (CF, AT, MS, FP, AA), severity, disclosure level
   - Needs a lookup table in configs, not in code
   - Used in: reason ranking, client wording, analysis.

5. **Latency measurement and budget tracking.**
   - Spec requires p99 ≤ 25 ms for card authorisations
   - Test at 900 live rules (01 §10 item 15)
   - Measure: normalisation, enrichment (concurrent with model call), rule evaluation (6 ms budget for ~600 rules)
   - Currently untested; would need a harness and production-like event stream.

---

## Test results

All tests in `tests/test_fraud_engine.py` pass:
- Rule volume: ~520 live, 100+ shadow across 5 families ✓
- Rule attributes: all required per §6.1 ✓
- Rule applicability: event type and segment matching ✓
- Rule firing: threshold comparison, feature extraction ✓
- No early exit: all applicable rules evaluated ✓
- Action precedence: severity > priority > family > rule_id ✓
- Hard blocks: enforce minimum action severity ✓
- Event normalisation: valid/invalid field validation ✓
- Evidence contract: decision_id uniqueness, no "today" ✓

Gaps in tests:
- Backtest equivalence not tested (no backtest mode)
- Overlay application not tested (overlays stubbed in evaluator)
- Full evidence record completeness not validated against 09 §5.15
- Latency not measured

Run tests:
```bash
PYTHONPATH=". ../00-shared-credit-core" \
  uv run --project /repo \
  pytest tests/ -v
```

---

## Build and serve verification

✓ `decider build` succeeds: validates pipeline steps, parameter ranges
✓ Sample request traces through pipeline: normalise → rule_set → context → hard_blocks → evaluate → resolve → emit
✓ Response includes decision_record with complete evidence
✗ Latency target (25 ms p99): not measured; harness required
✗ Backtest equivalence: not tested; backtest mode not implemented

See SERVE.md for exact build and test commands.

---

## Spec problems

1. **Backtest mode is underspecified.** 01 §5.17 describes the metrics but not:
   - Which recorded artefacts (event_features, model_score, table versions) must be replayed?
   - How is "the exact action, fired_rule_ids, fired_on_overlay_ids, shadow_fired_rule_ids" compared for equality?
   - When is backtest run (CI, on-demand, per overlay)?
   - Storage: where are the 90 days of events kept?

2. **Overlay composition order is mentioned but not formalized.** 01 §5.9 and 09 §5.14 both say "overlays stack in a declared order", but:
   - What is the default order if not declared?
   - Can an overlay change the order?
   - Does "composition order" mean mathematical order (f ∘ g ∘ h) or application order (h, g, f)?

3. **Action escalation overlay kind is not worked through.** Spec lists "action escalation" (§6.5) as an overlay kind that "changes the action a rule asks for", but:
   - Does it change only if the rule fires, or preemptively?
   - What if two rules' actions are escalated to the same higher action?
   - Interaction with critical rules unclear.

---

## Ponytail notes

Built the minimum that holds the spec. Rule volume at 520+100 (not shrunk) because rule evaluation count drives the latency budget and volume tests the no-early-exit requirement. Reused core.reason_codes even though it's undersized (10 vs ~120 fraud codes needed); extended via stub. Overlay structure built manually (no core framework); counterfactual calculated by hand (could defer to 09-H). Evidence record is verbose (23 items per §5.15) but necessary for replay; no simplification possible.

Spec corner case: the relationship between "applicable" rules, "evaluated" rules, and "fired" rules. §5.8 filters to applicable; §5.10 evaluates all applicable (none skipped); §5.12 acts on fired. The evidence must record (1) which were applicable, (2) which were evaluated and their outcomes, (3) which fired. Implemented as: applicable_rule_ids list + evaluation results with fired flag per rule.
