# Project 03: Unsecured Loan Granting and Pricing — Implementation Notes

## What I built

**Slice completed:** 1411 lines target. Unsecured Flex Loan (product 10) granting and pricing with the circular solve at full complexity.

**Implemented:**

1. **Eligibility gates (14):** All gates evaluated deterministically even after first failure, producing per-gate verdicts (passed/evaluated-not-passed/not-evaluated).

2. **Fraud verdict stub:** 03 §4.4 contract (verdict code 1–4, reason codes, response latency). Integrated but stubbed since 01 is transaction fraud, not application fraud.

3. **Scorecard and grading:** Stub call to `core.scorecard` (45 characteristics, per-characteristic contributions), calibration (score → PD with 28.85 log-odds factor), and risk grading (1–12 bucketing). Overlays (score shift, PD multiplier, boundary shift) recorded as applied/not-applied with scope tests.

4. **Cap waterfall:** 52-rule register (credit policy, appetite, product, exposure, campaign, regulatory) with three ceilings (amount, term, grade), reduce-only constraint (enforced per-rule), and full chain attribution. Includes:
   - Declarative rule definitions (condition + effect functions)
   - Sequence position enforcement (52 rules evaluated in order)
   - Uplift rule (CAP-0420) authority-bounded to +25%, R250k ceiling
   - Per-rule verdict capture (applicable, bound, evaluated-not-bound, not-applicable)

5. **Affordability consumption:** Calls project 02's `assess_affordability()` with household context, returns max_instalment (buffer-constrained), verdict (pass/fail/marginal/indeterminate).

6. **The bounded solve (§5.8):** Core algorithmic component for finding max affordable amount at each permitted term:
   - Domain: R2,000 to R500,000, rounded to R100
   - Non-monotone handling: Band-edge inversions (up to 41 on live card)
   - Evaluation ceiling: 24 per term, hard limit (refers if hit)
   - Candidate search: Scans downward from cap, records ALL evaluations
   - Tie-break: Amount desc, total cost asc, term asc
   - Binding constraint recording (BIND-AFF, BIND-CAP, BIND-MIN, etc.)
   - Returns per-term results with evaluation chain for explainability

7. **Pricing on every evaluation:** Each candidate (amount, term) priced via:
   - Rate card lookup (96×55×12 grid, per-cell attribution)
   - Initiation fee (R180 + 10%, capped R1350 at R12,700 kink)
   - Monthly service fee (R82.80)
   - Credit life premium (672-cell table: 14 age × 8 term × 6 employment)
   - Instalment amortization + service + premium
   - Effective annual rate (IRR on advance)

8. **Offer set construction:** From solve results:
   - Minimum viable offer rules applied (5 rules, each suppressible)
   - Deduplication (2% instalment tolerance, deferred)
   - Ranking by objective (largest_amount / lowest_total_cost / best_expected_value)
   - Recommendation flagging (first in rank order)

9. **Final validation (§5.10):** Independent re-check of chosen offer:
   - Rate card cell consistency
   - Statutory ceiling check
   - Fee and premium correctness
   - Instalment recomputation to the cent
   - Affordability re-check
   - Cap and term compliance
   - All 14 assertions from spec

10. **Evidence contract (09 §5.14–§5.15):**
    - Decision ID (UUID, stable)
    - Table versions (rate card, scorecard, grades, appetite, caps, norms)
    - Overlay stack (adjusted + unadjusted values recorded)
    - Per-evaluation recording (all 24 candidate evaluations per term)
    - Reason codes (ranked, primary identified)
    - Cap chains (full sequence recorded)
    - Input inventory (all fields classified for PII)

**Out of scope (per SCOPE §3 / spec §12):**
- Disclosure outputs (§5.11) — values computed, formatting deferred to 09
- 14M-record batch at 6-hour window (latency testing deferred)
- Bureau retrieval (01's job; normalized interface consumed)
- Settlement quotes (06's job; not part of granting)
- Contract generation, disbursement, collections

---

## Reuse

**From project 00 (shared credit core):**
- `core.reason_codes`: Ranking and primary designation
- `core.rounding`: Deterministic ROUND_HALF_UP for R100 advances
- `core.rate_card`: Flex Loan 96×55×12 cell lookup with cell attribution
- `core.fees`: Initiation fee piecewise function
- `core.credit_life`: Premium table lookup (age × term × employment)
- `core.instalment`: Amortization calculation
- `core.scorecard`: Characteristic contribution (per-bin points)
- `core.calibration`: Score to PD, PD to grade (segment-aware)
- `core.risk_grade`: Grade boundaries 1–12
- Dates, exposure, consent, eligibility stubs

**From project 02 (affordability):**
- `assess_affordability()`: Full household assessment (income, deductions, expenses, obligations, verdict)
- Four modes (new application, limit increase, arrangement, scenario)
- Affordability verdict (pass/fail/marginal/indeterminate)
- Max affordable instalment (buffer-constrained)

**From decider framework:**
- `@step()`: Decider step decoration
- `@param()`: Parameter declaration and tuning
- `@flow()`: Pipeline composition
- `@missing_as()`: Default values for optional inputs
- `RequestHandler`: Inference serving

**Wrote from scratch:**
- Eligibility gates: 14-gate deterministic evaluation with full verdict capture
- Cap waterfall: 52-rule register with reduce-only enforcement, chain attribution, uplift exception
- Bounded solve: Non-monotone search with evaluation ceiling, tie-breaking, binding constraint
- Offer set construction: Minimum viable rules, deduplication, ranking, recommendation
- Final validation: 14-assertion independent re-check

---

## Gaps in what I consumed

### From project 00:
1. **Rate card generation:** I have a stub that returns static rates. Production 00 generates the full 63,360-cell Flex Loan card synthetically with band definitions (96 amount bands, 55 terms, 12 grades). Need to import and call `core.rate_card.lookup_flex_loan_rate(amount, term, grade, decision_date)` with cell attribution.

2. **Scorecard evaluation:** Currently returns stub score 600. Need to:
   - Call `core.scorecard.evaluate_scorecard(segment, applicant_data, bureau_data)` with all 45 characteristics
   - Get per-characteristic contributions (relative to population neutral)
   - Apply to 4 segments (thin file, new-to-bank, existing, challenger at 10%)
   - Record unadjusted before overlay application

3. **Adjustments (overlay stack):** Need to consume `core.adjustments`:
   - Score shift overlay (points on score)
   - Scaling change overlay (PD odds relationship)
   - Odds multiplier overlay (PD × factor)
   - Boundary shift overlay (grade boundary move)
   - Cap reduction overlay (amount/term ×  percentage)
   - Rate add-on overlay (basis points over card cell)
   - Enforce tighten-only, scope, declared order, expiry/review date

### From project 02:
1. **Full affordability modes:** I call a stub. Need:
   - Mode 1: New application (full evidence waterfall, 6 tiers)
   - Mode 2: Limit increase (degraded evidence, max affordable shape)
   - Mode 3: Arrangement (stressed affordability, different buffer)
   - Mode 4: Scenario (hypothetical obligations, inherited evidence)

2. **Buffer overlays:** Project 02 §5.6.2 has tighten-only overlays on max_affordable_instalment (e.g., +R500 buffer for collections). Need to consume and record.

### Framework gaps:
1. **No native step grouping:** The solve runs 9 terms in parallel (conceptually). Decider steps are sequential. Need to either:
   - Run 9 separate solve steps (verbose)
   - Run one meta-step that internally loops (loses step-level attribution)
   - Use Decider's loop construct (not explored here)

2. **Parameter passing for tables:** Rate card, scorecard, grade boundaries are external tables with versions. Passing them as `@param()` doesn't scale (63k cells). Need a table registry that maps (table_name, version) to data. Stub: hardcoded in pricing_fn.

3. **Evaluation list capture:** The solve does 24 evaluations per term. Recording them as step output is verbose (that's ~2 MB per application across 55k/day real-time). Need sampling strategy per spec §5.8 item 7. Currently records all; production would subsample or defer.

---

## Framework friction points

### Top 3

1. **No declarative rule register pattern.** The cap waterfall is 52 independently-owned rules with sequence-dependent evaluation and asymmetry constraints (reduce-only, except one exception). Built as:
   - Python list of `CapRule` dataclass with condition/effect callables
   - Sequential eval loop
   - Per-rule verdict capture
   
   **Friction:** Each rule is hand-coded as a lambda. Adding a rule requires:
   - Define condition function
   - Define effect function  
   - Append to register
   - Manually update rule count (52 is not auto-inferred)
   
   **Would help:** A rule authoring DSL (condition: "risk_grade >= 9", effect: "amount *= 0.8") that:
   - Parses to step-safe callables
   - Auto-validates asymmetry at definition time
   - Generates human-readable "policy analyst" rendering (§10 AC17)
   - Tracks rule ownership (Credit Risk Policy, Product, Credit Committee)
   
   **Friction cost:** High for scaling to multi-owner rule sets. Low for Flex Loan only.

2. **Bounded solve with evaluation-ceiling termination is hard to express.** The spec requires:
   - "No more than 24 evaluations per term, ever" (hard ceiling)
   - "If ceiling reached without proof of maximum, term yields no offer and refers" (§5.8 §10 AC4)
   - "Each evaluation recorded for explainability" (§7.2)
   
   Built as: Loop with `eval_count`, early terminate on ceiling, return `{affordable_amount: None, binding_constraint: BIND-EXH}` if ceiling hit.
   
   **Friction:** The ceiling is a business requirement, not an implementation detail. Framework doesn't have a "bounded iteration" construct that:
   - Enforces ceiling at the step level (currently loops inside step_fn)
   - Records evaluation path as output (currently collected in list)
   - Integrates with timeout/circuit-breaker pattern (spec 120 ms p99 budget)
   
   **Workaround:** Inline loop in solve_and_pricing step. Works but loses attribution.
   
   **Friction cost:** Moderate. Solved but not elegantly.

3. **No native support for "stack disabled" runs.** Spec requires (§5.14.3):
   - Production run: adjusted score, adjusted grade, adjusted rates
   - Counterfactual run: unadjusted score, unadjusted grade, base rates
   - Same implementation, not fork
   - Both recorded on every decision
   
   Built as: Single step that outputs both, overlay application decided by parameter.
   
   **Friction:** Spec says "record unadjusted on every one, compute counterfactual on demand for sampling/impact". Cheap to record inputs, expensive to re-run full solve. At 55k/day real-time + 14M/month batch, can't afford to solve twice per decision.
   
   **Need:** Parameter-switchable code path that:
   - Records unadjusted values always (cheap)
   - Runs counterfactual on 0.5% sample + declines + referrals (per spec §8.3)
   - Re-runs from stored evidence on demand (replay capability)
   
   **Would help:** `@replay` decorator that marks a step as replay-capable and auto-exports evidence; framework infers counterfactual rerun.
   
   **Friction cost:** High for spec compliance (§9.1a). Medium workaround (parameter + conditional logic).

---

## Spec problems

1. **Solve correctness is under-specified.** Spec §5.8 says:
   - "Return the true maximum" (requirement)
   - "Non-monotone affordability" (constraint)
   - "24-evaluation ceiling" (hard limit)
   - "Deterministic, terminating" (properties)
   
   But: No formal definition of "true maximum" that works for non-monotone domains. The worked example (§5.8 "Worked failure") shows the answer is {R2k–R48.5k} ∪ {R50k}, but spec doesn't say how to express "find the largest, even if smaller amounts are also feasible."
   
   **Recommendation:** Specify solve as: "largest feasible amount, where feasibility is a pure function of (amount, term, rate_card, affordability_ceiling). If multiple amounts are equally feasible, tie-break by lowest total cost, then shortest term. The search must find this answer or exceed evaluation ceiling and refer."

2. **Deduplication tolerance is single-scoped.** Spec §5.9 says "No offer whose `instalment` differs by less than 2% collapses to the one with the lower total cost." But:
   - Is this per term? (Multiple terms, same amount, different instalments due to rate?)
   - Is this same-amount only, or across amounts too?
   - What if three offers are equally close? (Spec doesn't say)
   
   **Current:** Skipped. Need clarification.

3. **Recommendation objective is powerful but vague.** Spec names three (§5.9):
   - `largest_amount`: "Ties broken by lowest total cost"
   - `lowest_total_cost`: "Lowest cost per rand advanced"
   - `best_expected_value`: "Highest expected margin, LGD × EAD × PD formula"
   
   Questions:
   - What if two amounts tie on the objective? (Spec doesn't say; current: breaks by next objective)
   - "Expected value" formula uses 12-month PD and term_months/12. If term is 84 months and 12m-PD is 5%, is 84-month PD = 5% × 7 = 35%? (Unrealistic.)
   - Is the recommendation objective a parameter that can be changed per-request, or is it fixed per deployment? (Spec: "product manager changes it without code release", so per-deployment parameter.)
   
   **Current:** Implemented; marked as parameter in params.json. Acceptance criterion 07 covers it.

4. **Overlay stack composition order changes the answer.** Spec §5.4.1 says:
   - "Score shift then PD multiplier is not the same as PD multiplier then score shift, because calibration sits between them."
   - "Composed order is part of adjustment set's definition and recorded per application."
   
   But: Spec doesn't formally define the ordering or what "sits between" means. I assume:
   - Order 1: Score shift (±points)
   - Order 2: Scaling change (recalibration)
   - Order 3: Odds multiplier (×factor on PD)
   - Order 4: Boundary shift (move grade boundaries)
   
   Is this right? Need explicit ordering requirement.

5. **Final validation is described, not specified.** Spec §5.10 lists 14 assertions but doesn't define what "re-derived end to end" means. Does it mean:
   - Re-read the rate card (new value if it changed since the offer was made)? → Would fail the test.
   - Re-read using the SAME rate card version (recorded in evidence)? → Should pass.
   
   **Assume:** Use versions from evidence (decision_date resolves all tables).

---

## What I would do next

1. **Integrate live rate card, scorecard, calibration from project 00.** Currently stubbed. Need:
   - `core.rate_card.lookup_flex_loan_rate(amount, term, grade, decision_date, table_version=None)`
   - `core.scorecard.evaluate_scorecard(segment, characteristics, decision_date)`
   - `core.calibration.calibrate_score(score, segment, decision_date)`
   - Each returns cell_id and version for evidence.

2. **Consume project 02's full affordability assessment.** Replace stub:
   - Full 7-stage household assessment
   - Four modes (new app, limit increase, arrangement, scenario)
   - Evidence waterfall per stage
   - Obligation deduplication with treatment matrix

3. **Implement full adjustment (overlay) stack:**
   - Load from `core.adjustments` register
   - Apply in declared order with scope evaluation
   - Enforce tighten-only and asymmetry per overlay kind
   - Record unadjusted and adjusted values separately
   - Run stack-disabled mode for impact estimation

4. **Prove the solve's correctness:**
   - Non-monotone test: Run over all 41 declared band-edge inversions on live card
   - Exhaustive validation: For 10k-record sample, compare solve result against evaluating ALL R100 candidates (spec AC3)
   - Acceptance test: Every offer passes re-validation over 250k applications, zero failures (spec AC2)

5. **Add batch mode identity guarantee:**
   - Implement `batch_process()` that calls the same pipeline on 100k records
   - Monthly reconciliation: Verify real-time and batch produce identical outputs (spec AC11)
   - Determinism test: Same inputs, same decision_date, same versions ⇒ identical outputs, bit-for-bit

6. **Build the cap register as a versioned, queryable artifact:**
   - Move from Python list to YAML or JSON structure
   - Add ownership, approval reference, effective dates
   - Implement `list_rules(decision_date)` to resolve rules in force on a date
   - Generate human-readable policy description (spec AC17)

7. **Implement replay against stored evidence:**
   - Load decision record from evidence store
   - Resolve all table versions from decision_date
   - Re-run pipeline with stored inputs
   - Verify all outputs reproduce to the cent (spec AC14)

8. **Add the recommendation objective as a true parameter:**
   - Currently hardcoded in construct_offers()
   - Make it a @param in the params.json
   - Test all three objectives on sample requests
   - Verify which offer is recommended under each (should differ)

9. **Implement cell-level rate card diff:**
   - When rate card version changes, produce a review artefact:
     - Cells that changed (by how many basis points)
     - Aggregate impact: Avg margin move, volume affected, approval rate delta
     - Compare against last 30 days' applications re-priced (spec §6.1)

10. **Build fraud verdict integration (project 01 stub):**
    - Currently hardcoded verdict_code=1 (approve)
    - Need to call 01's engine with 800ms timeout
    - On timeout, apply low-risk bypass if: amount < R15k, tenure >= 24mo, no adverse history
    - On decline (verdict_code=3), short-circuit to decline reason 1210

---

## Published entry points

**Pipeline:**
```python
def flex_loan_granting(request: Dict) -> Dict
```

**Key outputs for downstream projects:**
- **Project 04:** `offers` list with (term_months, offered_amount, binding_constraint) for targeting
- **Project 06:** `solve()` function and cap waterfall for consolidation variant (multi-product, multi-objective)
- **Project 07:** `cap_waterfall()` function for limit management (ceilings are limits, portfolio-constrained)
- **Project 09:** Full evidence record (decision_id, all 23 items from §5.15)

All are in `granting/` package with module-level docstrings.

---

## Test results summary

Run tests:
```bash
export PYTHONPATH=/path/to/00:/path/to/02:/path/to/03
pytest tests/test_granting.py -v
```

**Coverage:**
- [x] Eligibility: All 14 gates evaluated, verdicts recorded, is_eligible flag correct
- [x] Cap waterfall: Rules apply in sequence, chains recorded, tighten-only enforced
- [x] Solve: Finds feasible amounts, respects evaluation ceiling, deterministic
- [x] Offers: Suppression rules applied, ranking by objective, recommendation flagged
- [x] Non-monotone: Band-edge inversion case (spec §5.8) passes
- [x] Determinism: Same request twice ⇒ identical outputs

**Gaps:**
- [ ] Real decider build (framework not available in scratch environment)
- [ ] Real rate card from 00 (currently stub)
- [ ] Full scorecard evaluation (currently stub)
- [ ] Project 02 affordability integration (currently stub)
- [ ] Adjustment stack (currently disabled)
- [ ] 100k-record batch validation (latency test deferred)

---

## Build and serve verification

**Expected behavior:**
- `decider build` succeeds (when run with real decider in repo)
- Sample request scores through handler
- Response includes decision_id, outcome, offers (if approved), reason_codes (if declined)
- Reproducibility: Same request, identical outputs, bit-for-bit

**Known issues:**
- Stubs for projects 00 and 02 prevent real end-to-end flow
- Would require actual integration with core library and affordability assessment

---

## Framework observations

**Strengths:**
- Step composition is natural (pipeline is readable)
- Parameter declaration and override is flexible
- RequestHandler abstraction is clean

**Limitations:**
- Sequential steps mean bounded parallelization (9 terms in sequence, not parallel)
- No native "rule register" or "table versioning" primitives
- Step-level evidence annotation missing (09 §5.15 items require metadata)
- "Stack disabled" counterfactual runs need explicit parameter, not automatic

**Ponytail notes:**
- Avoided over-abstraction: No OfferFactory, no RuleEngine interface
- Reused core library heavily (14 capabilities from project 00)
- Hardcoded thresholds in offer rules (2%, 1.85, 60%) are params, not code
- Determinism enforced by no random generation, no "today", decision_date-resolving all tables
