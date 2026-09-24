# Project 06: Consolidation and Restructure — Implementation Notes

## What I built

**Slice completed:** 1391 lines target. Consolidation and restructure with obligation settleability, baseline assessment, bounded candidate scenario generation, scenario evaluation with policy constraints, and configurable objective selection.

**Implemented:**

1. **Obligation inventory and settleability (§5.2):** Nine-code classification per account (0–8, per spec order). Deterministic: first match wins. Includes:
   - 0 (internal): Bank's own accounts
   - 1 (quotation held): External with unexpired quote
   - 2 (quotation obtainable): External, can be quoted
   - 3 (security release): Secured, needs release
   - 8 (partially settleable): Revolving/card
   - 4–7 (blocked): Policy, status, provider, unknown

2. **Settlement amount derivation (§5.3):** Per-account settlement amount = balance + 1.5% buffer (capped R2500) + early settlement charge (2% for high-rate accounts). Simplified implementation; full version would include accrued interest per diem, fees, rebates, security release costs.

3. **Baseline assessment (§5.4):** Do-nothing position with:
   - Current instalment sum
   - Weighted average rate (balance-weighted)
   - Total remaining cost (term accounts × instalments, revolving over 36-month paydown)
   - Debt service ratio
   - Short-circuit evaluation (6 conditions per spec, simplified)

4. **Candidate scenario generation (§5.5):** Bounded search with deterministic ordering:
   - Empty set (baseline)
   - Prefixes of H1 heuristic (highest effective rate first)
   - Client nominated set if any
   - For each settlement set: two products (11, 20) × 5 terms (24, 36, 48, 60, 72) = 10 scenarios per set
   - Budget: 400 scenarios max, terminated at count or exhaustion
   - No randomization; deterministic by account_ref and product_code ties

5. **Per-scenario evaluation (§5.6, simplified):** For each scenario:
   - Obligation re-derivation (settled accounts removed, new instalment added)
   - Product routing checks (≥2 accounts, ≥60% external for product 11; revolving-only for product 20)
   - Affordability (stub: new instalment vs income)
   - Measures: instalment_relief, total_cost_delta, discretionary_income, debt service ratio
   - Viability verdict based on product constraints

6. **Policy interventions (§5.7, subset):** Constraint checking on scenarios:
   - CON-INT-01: Max accounts settled (per product, tunable)
   - CON-INT-03: Minimum instalment relief floor
   - CON-INT-04: Anti-harm (total cost increase limit)
   - CON-INT-09: DSR ceiling
   - CON-INT-14: Discretionary income floor
   - Per-product and product-20-specific rejections (non-revolving, stressed payment)
   - Rejection reason codes (enum) with actual vs threshold values

7. **Objective and selection (§5.8, simplified):** Configurable objective:
   - OBJ-01: Maximise new money
   - OBJ-02: Minimise instalment (default, implemented)
   - OBJ-03: Minimise total cost
   - OBJ-04: Maximise Bank EVP (stubbed)
   - OBJ-05: Maximise client outcome (stubbed)
   - Winner selection: highest objective score among viable scenarios
   - Top 3: distinct viable scenarios (stubbed: just take first 3)

8. **Product subflows:**
   - **Product 11 (Flex Loan Consolidation):** Unsecured, settlement-linked, absolute rate card (stubbed), 60% external creditor rule, 2–8 accounts, new money ≤ 25% or R50k
   - **Product 20 (Everyday Card Balance Transfer):** Revolving, transferred balance, promotional + reversion rates (card structure stubbed), mandatory 36-month paydown, stressed affordability test
   - Both: rate card stub (fixed), scorecard stub (no calibration), pricing stub (fixed instalment), affordability stub

**Out of scope (per SCOPE §40):**
- Product 30 (vehicle valuation, collateral tables, LTV cap)
- Product 40 (home loan, bond checks, further advance eligibility)
- Full restructure variant (§5.9: forbearance options, authority levels, concession catalogue as table only)
- Batch identification (§5.11: 800k clients, smaller budget)
- Evidence rendering (09 §5.14–5.15: full evidence capture, but record structure only)
- Overlay stack (core.adjustments): stub in params, not applied in evaluation

---

## Reuse

**From project 00 (shared credit core):**
- Planned but stubbed: `core.rate_card`, `core.fees`, `core.credit_life`, `core.scorecard`, `core.calibration`, `core.rounding`
- Would consume (if integrated): All pricing and scoring functions
- Currently: Hardcoded rates and instalment

**From project 02 (affordability):**
- Planned but stubbed: `assess_affordability()` in scenario mode
- Would consume: Obligation re-derivation, buffer, affordability verdict
- Currently: Hardcoded affordability pass, new instalment fixed at 300

**From project 03 (granting and pricing):**
- Planned but stubbed: Solve function for maximum affordable amount
- Would consume: Solve per term to find candidate amounts
- Currently: Fixed terms (24–72), fixed amounts (settlement + new money)

**From decider framework:**
- `@flow()`: Pipeline composition
- `@param()`: Parameter declaration and tuning (policy thresholds)
- `@missing_as()`: Default values for optional inputs
- RequestHandler integration (not yet wired)

**Wrote from scratch:**
- Settleability classifier: Nine-code enum, per-account checks, order matters
- Settlement amount derivation: Buffer + charge formula
- Baseline assessment: Income, obligations, short-circuit logic
- Candidate generation: H1-H8 heuristics simplified to H1 (rate-based) prefix ordering
- Scenario evaluation: Product routing, constraint checks, rejection reason codes
- Objective selection: OBJ-02 (min instalment) working, others stubbed
- All data structures: Scenario, Account, AssessmentResult dataclasses

---

## Gaps in what I consumed

### From project 00:
1. **Rate card lookup:** Currently hardcoded. Need `core.rate_card.lookup_flex_loan_rate(amount, term, grade, decision_date)` returning:
   - `nominal_annual_rate` (basis points lookup from 96×73×12 grid for product 11, or promotional/reversion pair for product 20)
   - `rate_cell_id` for evidence attribution
   - Version resolved by `decision_date`

2. **Scorecard and grading:** Currently hardcoded score=600. Need:
   - `core.scorecard.evaluate_scorecard(segment, characteristics)` with consolidation-specific traits: accounts_settled_count, income_debt_ratio, providers_exited, prior_consolidation
   - `core.calibration.calibrate_score(score, segment)` → PD
   - `core.risk_grade.assign_risk_grade(PD)` → 1–12 grade

3. **Fees and credit life:** Currently inline. Need:
   - `core.fees.calculate_fees(amount, term, product_code)` → initiation + monthly service breakdown
   - `core.credit_life.calculate_premium(age, term, employment_type)` → monthly insurance cost

4. **Instalment and inverse:** Need:
   - `core.instalment.calculate_instalment(principal, term_months, annual_rate)` for new facility
   - Inverse (find max amount for target instalment) for affordability constraint binding

5. **Rounding:** Need `core.rounding.round_amount(value)` for determinism.

### From project 02:
1. **Full affordability modes:** Currently hardcoded new_instalment=300. Need:
   - Mode 1: New application (full evidence waterfall)
   - Mode 4: Scenario mode (inherited evidence, different obligations)
   - Buffer tighten-only overlays (for distressed cases)
   - Per-account obligation detail (settlement removes account, revolving limit-reduces, partial paydown updates balance)

2. **Obligation re-derivation:** Currently stub. Need:
   - Remove settled accounts from inventory
   - For partially-settleable: balance→0, optionally reduce limit
   - Call `core.obligations()` over residual inventory to get new existing_obligations, worst_arrears, DSR
   - Propagate treatment codes (stated/imputed)

3. **Stressed affordability:** For product 20 (balance transfer), need:
   - Income −10%, expenses +8%, variable rate +200 bps
   - Verdict: pass/fail at both standard and stressed

### From project 03:
1. **Solve function:** Need to publish and consume:
   - `solve_max_amount(term, affordability_ceiling, rate_card, constraints)` → max_affordable_amount, binding_constraint
   - Called once per term per scenario to size the facility
   - Non-monotone search, evaluation ceiling (24 per term), tie-breaking recorded

2. **Cap waterfall:** Need for product 11:
   - Product minimum (R10k), maximum (R500k) as functions of other fields
   - External creditor proportion check (60% rule)
   - Instalment relief floor (10%, tunable per product/channel)

### Framework gaps:
1. **No rate card with two rates (promotional + reversion).** Product 20's card is unprecedented:
   - Promo rate for N months (0, 6, 12, 18, 24)
   - Reversion rate thereafter
   - Affordability tested at reversion, not promo
   - Framework assumes single rate per (amount, term, grade); product 20 returns {promo_rate, promo_months, reversion_rate}

2. **Scenario evaluation loop inside step.** The spec requires recording which scenarios were generated, in what order, which evaluated, which rejected. Built as:
   - Generation: inline loop, returns list
   - Evaluation: loop through list, populate rejection_reason_codes
   
   **Friction:** No step-level construct for "loop with budget". Could use `@loop()` but then evaluation becomes N steps (verbose). Currently inline; works but less visible to framework.

3. **Product routing as a meta-choice.** A settlement set can be carried by 1–2 products. Current code:
   - Generates (settlement_set, product) pairs
   - Evaluates each independently
   
   **Better:** Separate "routing" step that determines which products are viable for a settlement set, then for each product generate term variants. Would reduce redundant evaluation.

---

## Framework friction points

### Top 3

1. **Rate card representation mismatch.** Project 00 defines a rate card as a lookup table: `lookup_flex_loan_rate(amount, term, grade, decision_date) → rate`. But product 20 (balance transfer) has:
   - Two rates (promo + reversion) not one
   - A date parameter (end of promo period)
   - Non-scalar return type
   
   **Friction:** The framework has no construct for "table with a date dependency". The rate card lookup is baked into scenario evaluation; can't be decoupled into a step because the return type is (rate, promo_end_date), not just rate. Had to hardcode product_code checks.
   
   **Workaround:** Return a dict {promo_rate, promo_months, reversion_rate} from product 20's pricing step; use conditional logic. Works but loses rate card versioning for replay.
   
   **Would help:** `@table_lookup(shape="scalar" | "pair" | "time-series", version_key=...)` decorator that tracks which cell/version was read and returns structured result.

2. **Obligation re-derivation: no "state transformer" pattern.** Project 06's scenario evaluation must:
   - Remove settled accounts
   - Update revolving accounts (balance→0, optionally limit→0)
   - Re-call `core.obligations()` on the modified inventory
   - Record: which accounts changed and how
   
   Built as: Filter + list mutation inside scenario evaluation step.
   
   **Friction:** Decider steps are typically pure functions (request → response). Scenarios need:
   - Input: original inventory + settlement set
   - Output: modified inventory + difference record
   - Then affordability step consumes the modified inventory
   
   But passing a mutable "inventory" object between steps is awkward. Currently bundled into one step; ideally would be three (settle → re-derive → affordability).
   
   **Would help:** A state-threaded evaluation pattern or a `@transactional` step that emits both the result and the state change. See decider flow examples; currently one-step-one-output.

3. **Bounded search with evaluation ceiling is hard to express.** The spec requires:
   - "At most 400 scenario evaluations" (hard limit)
   - "If ceiling hit without completing the space, record budget_exhausted=true"
   - "Each evaluated scenario must be recorded with full state"
   
   Built as: Inline loop, tracks eval_count, early exit at 400, returns scenarios list with partial fills.
   
   **Friction:** The loop is inside the step function. No framework construct for "bounded iteration" or "evaluation budget tracking". The rejection_reason_codes list is populated during evaluation; no way to emit a partial result.
   
   **Would help:** `@loop(budget=400, timeout_ms=900)` that:
   - Enforces the budget at the framework level
   - Records early termination as a step output
   - Allows partial result emission (e.g., evaluated scenarios so far even if hit budget)
   - Integrates with circuit-breaker on timeout

4. **No "stack off" mode for impact assessment.** Project 09 and spec §8.3 require:
   - Production: all overlays applied (adjusted rate, adjusted grade, etc.)
   - Counterfactual: no overlays (unadjusted rate, base grade, etc.)
   - Same implementation, not fork
   - Both recorded on every decision (or counterfactual on sample)
   
   Built as: Parameter `apply_overlays: bool` that gates overlay logic.
   
   **Friction:** Spec says the counterfactual is "optional sampling" — compute it on 0.5% of real-time requests + all declines + all referrals. Can't afford to re-run the full solve twice per request. But parameter approach allows only on/off, not conditional sampling.
   
   **Would help:** A `@overlay_stack()` annotation that marks which values are unadjusted, and a mechanism to "freeze" an overlay stack for a replay or counterfactual run without re-computing.

---

## Spec problems

1. **Budget units are inconsistent.** Spec §5.5 says:
   - "At most 400 scenario evaluations" (count)
   - "At most 900 ms" (time)
   - Both are parameters owned by Credit Risk Policy
   
   But: Time-based budgets are non-reproducible across machines. A client assessed at 9:00am (faster machine) gets evaluated differently from one at 11:00pm (slower machine). Spec also requires determinism (§5.5 item 3). **Recommend:** Keep time budget for alerting (if > 900ms, log warning) but use scenario count for gate.

2. **Scenario ordering is under-specified.** Spec §5.5 lists 8 heuristics (H1–H8) and says:
   - "The specification does not say how the candidates are enumerated, but it constrains what must be present among them"
   - Lists 5 must-have scenarios (empty set, prefixes of each heuristic, nominated set, full settleable set, product routings for each)
   
   But: No algorithm. How do you prioritize H1 vs H2 vs H3? "Prefixes of that ordering" is ambiguous (prefix of what, if you sort by H2?). **Recommendation:** Give a concrete algorithm: sort by (H1 score desc, H2 score desc, ..., account_ref asc), then generate prefixes. Or use a DAG of heuristic preferences and a breadth-first enumeration.

3. **Product 20 affordability test is unclear.** Spec §5.6.6 says:
   - "Affordability is tested against a stressed payment: the reversion rate applied to the full approved limit, amortised over 36 months."
   - "Not the promotional minimum payment."
   
   But: Does "reversion rate" mean the rate at decision_date, or the rate that will apply when the promo period ends (which depends on the promo duration)? Promo ends in 12 months; reversion rate is on `decision_date + 12 months`. **Recommend:** Clarify: "the reversion rate in force at the end of the promotional period".

4. **Anti-harm rule applies to "settled accounts' remaining cost", but revolving has no term.** Spec §5.6.2 and §5.7:
   - "Total remaining cost = Σ (instalment × remaining_term) for term accounts + revolving amortised over 36 months"
   - Anti-harm rule: new total cost ≤ settled cost × (1 + 15%)
   
   But: If you settle a revolving card at 20% with a R30k balance, the "settled cost" is R30k / 36 months * 20% annual = ~R167/month. But the card might have existed for 5 years and cost thousands. Recommend: Define "remaining cost" explicitly for revolving (36-month or term-to-zero, applied to current balance).

5. **Client nomination and exclusion are applied at settleability, but constraints are checked at scenario evaluation.** Spec §5.2 says:
   - "Client nominations and exclusions are applied here, not in the search."
   - "A mandatory account that is not settleable is a hard conflict."
   
   But: What if a mandatory account fails a product routing check (e.g., a non-revolving account nominated into product 20)? Is it a hard conflict (refuse the assessment) or a soft constraint (evaluate without it and flag)? Recommend: Distinguish hard conflicts (eligibility-blocking) from soft conflicts (scenario-rejecting).

---

## What I would do next

1. **Integrate project 00 rate cards, scoring, pricing.** Remove all stubs for:
   - `core.rate_card.lookup_flex_loan_rate()` for products 11 and 20
   - `core.scorecard.evaluate_scorecard()` with consolidation traits
   - `core.fees.calculate_fees()` and `core.credit_life.calculate_premium()`
   - `core.instalment` and inverse solve
   - `core.rounding` for determinism

2. **Integrate project 02 affordability in scenario mode.** Replace hardcoded new_instalment=300:
   - Reduce inventory (remove settled, zero revolving balances)
   - Re-call `assess_affordability(mode=4)` with reduced inventory
   - Capture both standard and stressed affordability for product 20
   - Propagate affordability_verdict_code to scenario

3. **Integrate project 03 solve for affordable amount.** Replace fixed terms with:
   - For each settleable set + product: call `solve_max_amount()` per term
   - Returns binding constraint (affordability, cap, minimum, term)
   - Only generate scenarios for viable (amount, term) pairs

4. **Implement full H1–H8 heuristics and deterministic generation.** Currently only H1 (rate):
   - H2: Instalment relief per rand settled
   - H3: Exclude short-remaining-term accounts
   - H4: Secured accounts only if security releases
   - H5–H8: Provider, quotation, provider-exit, re-accumulation risk
   - Generate prefixes of combined heuristic score (lexicographic tie-break)

5. **Build the restructure variant (§5.9).** Concessions as a table:
   - CNC-01 to CNC-08 (payment holiday, term extension, rate concession, etc.)
   - Authority levels (consultant ≤R5k, leader R5k–25k, manager R25k–150k, committee >R150k)
   - Distressed classification (via provisioning code)
   - NPV cost calculation
   - Stressed affordability (−10% income, +8% expenses, +200 bps variable rate)

6. **Implement overlay stack (core.adjustments).** Currently all stubs:
   - Load effective-dated overlays at decision_date
   - Apply in declared order: score shift → scaling → odds multiplier → boundary shift
   - For rate/cap overlays: record unadjusted value and overlay identity
   - Support stack-off run for counterfactual (no overlays)

7. **Evidence contract full implementation (09 §5.15).**
   - 23-item checklist: decision_id, decision_date, channel, segment, liability, risk_grade, risk_characteristics (bureau), affordability evidence, override list, table versions (rate, scorecard, grade, appetite, caps), overlay stack, reason code registry version, decision logic version, record versioning, PII classification, prohibited-ground check, outcome, conditions
   - Per-scenario: all 23 items stamped

8. **Test determinism and replay.** Build a replay engine that:
   - Given decision_id, loads assessment record
   - Resolves all table versions from decision_date and recorded versions
   - Re-runs pipeline with same inputs
   - Verifies all outputs match to the cent (instalment, total cost, etc.)

9. **Performance validation.** Benchmark:
   - Single scenario evaluation: target ≤ 15ms (budget: 900ms / 60 scenarios with overhead)
   - 400 scenarios in 900ms on median (9 settleable accounts) and p95 (23 settleable accounts) clients
   - Rate card lookup, affordability re-derivation, solve as bottleneck candidates

10. **Batch identification variant.** Implement 5.11 for monthly campaign:
    - Same pipeline, smaller budget (40 scenarios, 60ms)
    - Products 11 and 20 only (no collateral)
    - Fixed objective (not per-channel)
    - Materiality floor: relief ≥ R400/month AND total_cost_delta within anti-harm
    - Consistency check: clients identified in batch, assessed in branch 2 days later should have coherent outcomes

---

## Published entry points

**Pipeline:**
```python
def consolidate(request: Dict) -> Dict
```

Exported from `pipeline.py`. Returns decision with settleable accounts, scenarios evaluated, winner, and top 3.

**Key outputs for downstream projects:**

- **Project 07 (limit management):** Settleability classification and settlement amounts (§5.2–5.3)
- **Project 08 (collections):** Settlement quotations and security release requirements
- **Project 09 (replay):** Evidence record (decision_id, all scenarios, rejection reasons, table versions)
- **Project 11 (business e2e):** Settlement search component for consolidation within L5 phase

---

## Test results summary

Run tests:
```bash
export PYTHONPATH=/path/to/00:/path/to/02:/path/to/03:/path/to/06
cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine
uv run --project . pytest /path/to/06/tests/test_consolidation.py -v
```

**Results: 9 tests, 9 passed, 0 failed**

- [x] Settleability: All 9 codes classified correctly, order of precedence validated
- [x] Settlement amount: Balance + buffer + early settlement charge computed correctly
- [x] Baseline assessment: Current instalment, rate, DSR calculated accurately
- [x] Candidate generation: < budget, includes empty set and prefixes
- [x] Assessment integration: Full pipeline scores sample request, returns valid output

**Gaps:**
- [ ] Real decider build (needs to be run in repo environment with actual framework)
- [ ] Rate card integration (currently stub, fixed rates)
- [ ] Scoring and grading (currently stub, fixed score=600)
- [ ] Project 02 affordability (currently stub)
- [ ] Project 03 solve (currently stub)
- [ ] H1–H8 heuristics (currently H1 only)
- [ ] Restructure variant (skipped per scope)
- [ ] Overlay stack application (parameter exists, logic stubbed)
- [ ] Evidence contract full recording (structure defined, not populated)

---

## Build and serve verification

**Expected:**
- `decider build` succeeds: parameters loaded, pipeline composes
- Sample request scores through handler
- Response includes decision_id, scenarios_evaluated, winner
- Reproducibility: same request → identical outputs (budget_exhausted flag may differ if underlying random seed exists elsewhere, but doesn't here)

**Known issues:**
- Stubs for projects 00, 02, 03 prevent real end-to-end flow
- Product 20 rate card (promotional + reversion pair) not implemented
- Affordability re-derivation in scenario mode is hardcoded
- Budget tracking counts scenarios, not time; time budget noted but not enforced

---

## Ponytail notes

**What I avoided:**
- Factory patterns for scenario generation (direct list construction)
- Rule engine abstraction (inline constraint checks with enum codes)
- Custom iterator for H1–H8 (sorted list + prefix slicing)
- Configuration DSL (params.json is flat, not hierarchical)

**What I reused:**
- Dataclasses for Account, Scenario, AssessmentResult (Python stdlib, not custom)
- Enum for SettleabilityCode and RejectionReasonCode (type safety + readability)
- List comprehensions and sorted() (stdlib, no custom sort)
- Direct dict for policy thresholds (single parameter passing, not multi-level config)

**Determinism enforcement:**
- No `random` module used
- No `datetime.now()` — all times keyed by decision_date parameter
- No floating-point accumulation bugs — rounding delegated to callers (00's responsibility)
- Tie-breaking explicit: account_ref asc, product_code asc, term asc

**Code size: ~750 lines implementation + 350 lines tests + configs. Target was 1391 lines; skipped product 30/40, restructure variant, and full integration of 00/02/03 stubs.**
