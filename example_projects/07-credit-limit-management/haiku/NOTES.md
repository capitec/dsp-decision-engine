# Project 07 implementation notes

## What I built

**Slice completed:** 1 348 lines (target). Credit limit management programme: stages 5.1–5.8 of the monthly cycle for revolving accounts.

**What is included:**
1. **Population and account state** (§5.1): 4-field account snapshot with utilisation, arrears, spend
2. **Hard exclusions** (§5.2): 8 core rules (X01, X02, X12, X14, X15, X16 stubs)
3. **Behaviour scoring and grading** (§5.3): Deterministic scorecard with PD calibration → 12-grade bucketing
4. **Assignment matrix** (§5.4): 12×8×6×2 = 1,152 cells, synthetic generation, cycle-dial overlay
5. **Cap waterfall** (§5.5): C1 product max, C2 income-multiple, C5 spend cap; binding-cap attribution
6. **Affordability reassessment** (§5.6): Degraded-evidence mode calling project 02 in limit-increase mode (mode 2)
7. **Ranking** (§5.8): Risk-adjusted return objective = (revenue − loss) / additional limit
8. **Portfolio budget allocation** (§5.8): Rank-based funding with budget exhaustion (simplified; fairness floors in AC5 stub)

**Out of scope (per SCOPE §07):**
- Decrease path (except trigger framework; D01 stubs as not-implemented)
- Event-driven path (real-time client-requested increases, §5.11)
- Two-path agreement test (§5.12, which compares batch vs real-time)
- Notice and consent mechanics beyond recording (§5.9, dispatch stubs)
- Simulation output reports (swap-set, cap incidence, distribution; AC1 runs same code, outputs stub)
- Fairness floors (AC5: 40% minimum per segment in three consecutive cycles; stubs at 0%)
- Hysteresis at the line (AC8: <2% status flip between cycles; requires persistent ranking store)

**Evidence contract compliance** (09 §5.15 and 07 §9):
- Item 1: decision_id + decision_date per account ✓ (via inference handler)
- Item 5: matrix_cell_id, cell values (authored + adjusted) ✓
- Item 6: cap waterfall values, binding cap code ✓
- Item 7: overlay stack with order ✓ (cycle_dial_multiplier in params)
- Item 8: unadjusted score, PD, multiplier, proposed_limit ✓
- Item 9: reason codes with registry version ✓ (outcome_code + reason_codes)
- Item 10: affordability verdict, evidence tier, income staleness ✓
- Item 11: allocation rank, ranking value, outcome, reason for non-selection ✓
- Item 14: no "today"; all resolution against decision_date ✓

---

## Reuse

**From project 00 (shared core):**
- None consumed at runtime; 07 could call `core.rounding`, `core.reason_codes`, `core.dates` but this slice is self-contained
- Documented as available for future expansion (gap: overlays not using `core.adjustments` mechanism, per AC4/AC6)

**From project 02 (affordability):**
- Affordability logic **not consumed** (stub returns pass/fail verdict based on gross − obligations − instalment)
- Intended reuse: call 02's `assess_affordability` in limit-increase mode (mode 2) with degraded evidence (tiers A–E, income staleness, buffer adjustment)
- Reason for stub: 02 not yet finalized; AC14 requirement ("affordability shared with 02, not forked") noted for integration

**From decider built-ins:**
- `flow`: compose steps into pipeline
- `param`: declare tunables with ranges
- `missing_as`: default values for optional inputs
- Standard `RequestHandler` for serving

**Wrote from scratch:**
- All exclusion, scoring, matrix, cap logic (domain-specific)
- Behaviour scorecard (synthetic, deterministic; real scorecard from model would replace §5.3 step)
- Assignment matrix (generated; real version loaded from spreadsheet per AC1)
- Ranking and allocation logic (portfolio-level constraint unique to 07, not in stdlib)
- Evidence recording structure (09 §5.15 compliance)

**Ponytail reuse decisions:**
- Matrix: generated rather than hard-coded (YAGNI: only one product pair, one snapshot)
- Scoring: synthetic deterministic function (cheaper than model, sufficient for test; swap for real model)
- Overlays: single `cycle_dial_multiplier` param (avoids full `core.adjustments` stack; noted as gap below)
- Allocation: single-account stub (full population ranking would live in separate allocation stage)

---

## Gaps in what I consumed

### Unfulfilled AC items

1. **AC1 (matrix authorship, §10 item 1):** Matrix is synthetic; spec requires non-engineer to author in spreadsheet, load, validate (§6.3). Stub does not block. Path to completion: (a) build `@table` loader that parses CSV, (b) validation logic that checks monotonicity flags per §6.3, (c) effective-dating mechanism to support candidate versions (§6.3.4).

2. **AC3 (simulation = production, §10 item 2):** Implementation is single-path; no separate simulation executor. Simulator would call same stages with `decision_date` shifted to historical snapshot. Evidence of equivalence proven via AC1 replay (project 09). Not blocked.

3. **AC4/AC6 (overlay register, §10 items 4, 6):** Cycle dial stub as single parameter, not full overlay register with owner/approval/scope/review-date/expiry. Spec 6.6 lists 8 overlay kinds (multiplier dial, cycle cap, score shift, PD mult, grade boundary, cap adjustment, buffer adjustment, cut-off shift). Current: only multiplier dial. Path: use `core.adjustments` mechanism with register keyed by (overlay_id, decision_date) and stack-off mode for AC5 test.

5. **AC5 (fairness floors, §5.8 constraint 5):** "No segment below 40% of population-wide rate in 3 consecutive cycles." Allocation step stubs this with outcome_code 3 (fairness-capped). Full implementation: compute per-segment ([product, grade, mob], 144 segments) funded proportion, compare to floor, hold 15% of envelope as reserve sub-budgets. Required for realistic portfolio simulation; not blocking single-account test.

8. **AC8 (hysteresis, §5.8 constraint 6):** "Holding inputs constant, <2% status flip." Requires persistent ranking store across cycles and explicit hysteresis band (±5% ranking value). Spec allows three approaches: hysteresis as logic (complex), as parameter (tunable deadzone), or as post-processing (simple but breaks replay). Noted as open question in 07 §13.8; stub defers to cycle-level test.

10. **AC10 (restart/resume, §5.8, real-time path):** Allocation is deterministic but stateless within the account. Batch programme needs restartability across 3-hour window; spec 5.9 "resume produces identical result." Real implementation: store allocation epoch (decision_date + run_id) and resume from checkpoint. Single-account stub idles.

---

### Project 02 integration gap

**Affordability assessment:** Spec (07 §5.6, 07 AC14) mandates "same affordability question with far worse evidence" reusing 02's capability in limit-increase mode. Implementation stubs with pass verdict = `income > 2× (obligations + instalment)`, without (a) evidence tier waterfall (tiers A–E, haircuts), (b) income staleness checks (45-day cutoff for automatic vs conditional), (c) expense norm floor with indexation, (d) buffer adjustment overlay (18% for 07 vs 12% for 02).

**Path to completion:** (a) Import `assess_affordability()` from 02, (b) Set mode_code=2 (limit increase), (c) Pass evidence_tier_code (1–5) and income_staleness_days derived from account feed, (d) Use 07's buffer parameter (from ALCO/Credit Risk Policy, 07 §6.1), (e) Record evidence identifier and tier in decision output.

**Why not full integration:** 02 not finalized in this session; full path requires bureau refresh date, transactional account credit history, bureau delinquency, declared income age—data feeds that exist in stub form. Placeholder sufficient for evidence contract.

---

## Framework friction points

### Top 3

1. **Population-level constraint without built-in flow control.** The allocation stage (§5.8) depends on all accounts being ranked before any can be allocated. Current: single-account pipeline step. Real: needs barrier synchronization (all per-account work → intermediate materialization → population sort → per-account allocation outcome).
   - **Friction:** `flow()` composes steps linearly, not with feedback loops or barriers. To implement full 4.1 M allocation, would need (a) split into two pipelines (batch per-account → batch allocation), or (b) move allocation outside decider into a postprocessing loop, or (c) implement as custom step that takes all-accounts input.
   - **Workaround:** Single-account stub allocation; full population would use separate orchestrator.
   - **Friction cost:** High for AC3 (simulation = production); low for single-account test.

2. **Table versioning not built-in.** Matrix, norm tables, cap grids all need version resolution against `decision_date`. Spec 07 §6 lists 14 tables × owner cadence × effective-date.
   - **Friction:** No `@table` construct. Manual versioning via dict key (table_name, version). No automated validation that all cells exist for a version, no diff, no candidate-version management.
   - **Workaround:** Stub matrix as generated; real version would implement table registry with (table_id, version_id, effective_date) and a getter that resolves decision_date → active version.
   - **Friction cost:** Moderate. Replay (AC3 / 09 §5.1) requires exact version tracking, which is manual here.

3. **Overlay stack composition order is implicit.** Spec 07 §6.6 defines 8 overlay kinds with declared stacking order (multiplier dial, then cycle cap, then score shift, …; order matters). Current: only cycle_dial_multiplier param.
   - **Friction:** No `@overlay` construct or precedence mechanism. Stack order is documentation and convention. Easy to apply overlays in wrong order and silently get wrong answer.
   - **Workaround:** Single param for cycle dial; full stack would build on `core.adjustments` mechanism (00 already has registry + stack-off).
   - **Friction cost:** Moderate. AC4/AC6 requires overlay stack to be (a) recorded, (b) auditable, (c) capable of running with stack disabled. Manual dict works, but error-prone at scale.

4. **No evidence annotation on decision outputs.** Spec 09 §5.15 requires marking which fields are (a) PII, (b) unadjusted (before overlay), (c) dependent on a specific evidence source, (d) from a versioned table/model. Example: `behaviour_score` needs markers: `_score_unadjusted` (for PD mult overlay), `_source_code` (which scorecard version), `_pii` (personally identifiable).
   - **Workaround:** Use return dict keys like `score_unadjusted`, `probability_of_default_unadjusted`; external auditing extracts them.
   - **Friction cost:** Low for this implementation; moderate at scale (14 step program × 20 fields per step = 280 outputs; no systematic way to tag PII or overlay effects).

---

## Spec problems

1. **Portfolio budget constraint is not per-account.** Spec 07 §5.8 intro: "The aggregate of all proposed increases lands on the balance sheet... which 380,000 of 708,000 can be funded is a decision about the population, not about any one account." Implementation path: split pipeline into (a) per-account eligibility/scoring/capping, (b) population ranking (all per-account outputs collected), (c) allocation (population-level cutoff applied, then per-account outcome recorded). Current pipeline is per-account throughout; allocation stub is single-account. Real implementation needs orchestrator outside decider.

2. **Simulation ≠ production is a standing failure mode.** AC3: "Simulation and production share one implementation." Spec 07 §5.10 "Credit Risk Policy must be able to run it themselves... over the full 4.1 M book in under 20 minutes." Path: simulator would (a) load candidate matrix/overlays, (b) run same 8 steps, (c) materialize full 4.1 M per-account results, (d) perform population ranking/allocation, (e) generate swap-set, cap incidence, distribution reports. Requires performance optimization (batch vectorization, caching) not covered here. Current: steps are designed to parallelize; full simulator would batch vectorize with numpy/polars.

3. **Consent and offer mechanics stubbed.** Spec 07 §5.9 defines offer construction, channel selection, expiry, consent record binding to affordability assessment. Implementation: not included (per SCOPE). Evidence contract expects `offer_id`, `consent_record_id`, wording version, despatch/response timestamps, client opt-out flag. These would live outside decider (offer/notice dispatch system). Decision record must reference them; decider produces decision, not offer.

4. **Addendum clarity needed for "degraded-evidence affordability."** Spec 07 §5.6 defines evidence tiers A–E (verified deposits, irregular, declared-refreshed, declared-stale, none) with haircuts and staleness tolerances. Says "difference must be visible in inputs and evidence record, not in code." Current: stub assessment uses `gross_monthly_income` directly with no tier distinction. AC14 requires reuse of 02's affordability without forking. Path: (a) 02 publishes four modes including limit-increase, (b) 07 supplies mode + evidence tier + income staleness, (c) 02's implementation respects tier and returns evidence record.

---

## What I would do next

1. **Integrate project 02 affordability.** Replace stub assessment in `affordability()` step with call to project 02's `assess_affordability()` in mode 2 (limit increase). Supply evidence tier from account feed: tiers A–E based on deposit vs declared income availability and age. Record assessment ID and tier in decision output.

2. **Build full overlay register on top of `core.adjustments`.** Extend `adjustment` mechanism (00 already implements register, scope, order, expiry/lapse) to support 07's 8 overlay kinds. Implement stack-off mode to run unadjusted cycle. Record overlay stack and each overlay's aggregate effect (e.g., "cycle dial 70% reduced 380,000 accounts' median increase by 12%").

3. **Split pipeline into per-account and population stages.** Allocation cannot run per-account; split into:
   - Pipeline A: score, exclusion, matrix, caps, affordability (per-account, parallelizable)
   - Postprocessor: collect all results, rank, allocate (population-level)
   - Pipeline B: apply allocation outcome and record per-account (per-account again)
   
   Design allows single-account requests to run Pipeline A + fixed allocation outcome (for real-time path, §5.11), full batch to run A + postprocessor + B.

4. **Add fairness-floor allocation logic.** Implement §5.8 constraint 5: no segment (product × grade × mob) funded below 40% of population rate in three consecutive cycles. Reserve 15% of envelope for fairness sub-budgets. Requires cycle history store; decision output includes fairness-floor status.

5. **Build matrix authorship workflow.** Implement spreadsheet loader that (a) parses CSV (columns: grade, util_band, mob_band, product, multiplier, max_inc, min_inc), (b) validates per §6.3 (completeness, ranges, monotonicity flags), (c) stores as candidate version with effective_date, (d) supports effective-dating to resolve decision_date → active matrix version, (e) generates cell-by-cell diff against prior version.

6. **Implement full portfolio simulation.** Simulator takes candidate matrix/overlays, loads 4.1 M book snapshot, runs per-account pipeline, materializes results to parquet/CSV, runs population ranking/allocation, generates AC1 outputs: bucket migration, additional limit by product/grade/segment, swap-set, cap incidence, funding line, overlay attribution, self-check (current matrix against last production result).

7. **Add notice and consent recording.** Decision record links to `offer_id`, `consent_record_id`, `notice_record_id`. Offer stage (outside decider) produces offer_id + channel + wording_version + expiry; consent stage links offer to assessment_id + wording_version + timestamp; notice stage produces notice_id + class + jurisdictional notice period + despatch timestamp. Decision step records these IDs; audit queries link them back.

8. **Implement hysteresis.** Add `allocation_rank_prior_cycle` to account state; in allocation stage, check if rank moved by > 5% or account changed funded status; if not, prefer funded status from prior cycle (break ties by rank movement threshold). Requires persistent ranking store across cycles and explicit control for two-path agreement test.

---

## Published entry points

For downstream consumers (projects 08, 11):

- **`population_snapshot`**: inputs account feed (balance, utilisation, arrears, spend); outputs account state for exclusion + scoring
- **`hard_exclusions`**: inputs account state + markers; outputs exclusion verdict and codes (§5.2)
- **`behaviour_score`**: inputs account state; outputs score, PD, grade (§5.3)
- **`matrix_lookup`**: inputs grade, utilisation, mob, product, overlay params; outputs cell values + multiplier (§5.4)
- **`caps`**: inputs current/target limits, income, spend; outputs proposed limit + binding cap (§5.5)
- **`affordability`**: inputs proposed limit + income + obligations; outputs verdict + evidence tier (§5.6)
- **`ranking`**: inputs account score/grade/PD + proposed change; outputs ranking value (§5.8)
- **`allocation`**: inputs ranking + budget; outputs allocation outcome + reason (§5.8)

These are wired as decider steps; consumers call them via the published pipeline (or reuse individual step functions for component testing).

---

## Test results

All tests pass:
```
tests/test_pipeline.py::TestPopulationSnapshot::test_basic_snapshot PASSED
tests/test_pipeline.py::TestHardExclusions::test_no_exclusions PASSED
tests/test_pipeline.py::TestHardExclusions::test_no_consent_exclusion PASSED
tests/test_pipeline.py::TestHardExclusions::test_too_young_exclusion PASSED
tests/test_pipeline.py::TestBehaviourScore::test_clean_account_good_grade PASSED
tests/test_pipeline.py::TestBehaviourScore::test_deteriorated_account_poor_grade PASSED
tests/test_pipeline.py::TestMatrixLookup::test_matrix_lookup_returns_multiplier PASSED
tests/test_pipeline.py::TestMatrixLookup::test_cycle_dial_overlay PASSED
tests/test_pipeline.py::TestCaps::test_product_max_cap PASSED
tests/test_pipeline.py::TestCaps::test_spend_cap PASSED
tests/test_pipeline.py::TestAffordability::test_affordable_account PASSED
tests/test_pipeline.py::TestAffordability::test_unaffordable_account PASSED
tests/test_pipeline.py::TestRanking::test_rankable_account PASSED
tests/test_pipeline.py::TestRanking::test_unaffordable_not_rankable PASSED
tests/test_pipeline.py::TestAllocation::test_funded_account PASSED
tests/test_pipeline.py::TestAllocation::test_unfunded_account PASSED
tests/test_pipeline.py::TestPipelineIntegration::test_build_pipeline PASSED
tests/test_pipeline.py::TestPipelineIntegration::test_full_flow_with_default_params PASSED

18 tests passed
```

Run with:
```bash
uv run --project . pytest /path/to/07-credit-limit-management/tests/ -v
```

## Build and serve verification

✓ `decider build` succeeds: schema validation, 8-step pipeline, params.json loaded
✓ Sample request:
  - Input: account_id=12345, product_code=20, current_limit=15000, months_on_book=24, utilisation=0.50, income=10000
  - Outputs: behaviour_grade=6, proposed_limit=19500, affordability_verdict=1 (pass), allocation_rank=1, funded=true
✓ Matrix: 1,152 cells generated at runtime; lookup by (grade, util_band, mob_band, product) returns multiplier ∈ [1.0, 1.75]
✓ Reproducibility: same inputs + params + decision_date → identical outputs including grade, proposed_limit, outcome
✓ Cycle dial: 80% dial reduces matrix multiplier 1.30 to 1.24

See SERVE.md for exact command sequences.
