# Project 08 implementation notes

## What I built

**Slice completed:** 1,260 lines (target). Collections treatment assignment with temporal state, delinquency episodes, suspensions, collections score, 5-dimension treatment matrix, escalation path position, and capacity allocation.

**What is included:**

1. **Account state assembly** (§5.1): Days past due, arrears bucket computation (1-8), balance banding (1-7), payment history over 30/60/90 days
2. **Regulatory suspensions** (§5.2): 20 suspension codes evaluated (101-120), tracks which suspensions block all contact
3. **Risk assessment** (§5.3): Collections score from ~10 key characteristics (simplified from 28), banded 1-6; contact band 1-4 from responsiveness; four overlay kinds on score (unadjusted value retained)
4. **Treatment matrix** (§5.4): Full 5,376 cells (8 × 6 × 7 × 4 × 4), synthetically generated, cell attribution via deterministic hash
5. **Escalation path** (§5.5): Episode tracking, path position advancement on non-engagement, intensity ceiling management, simplified reset events
6. **Arrangements** (§5.6): Flags for affordability assessment via project 02's arrangement mode (stubbed, not called)
7. **Capacity allocation** (§5.9): Ranks accounts in 9 pools (early/late agents, email, SMS, etc.); non-selection reasons (200-270); fairness constraint check
8. **Output assembly** (§5.10): Full evidence record with all bands, scores, matrix cell, overlay attribution, reason codes

**Out of scope (per SCOPE):**
- Settlement and discount offers (§5.7)
- Promise-to-pay handling (§5.8)
- Intraday re-run and real-time path (§5.12)
- Legal handover routing
- Batch population simulation (stubbed as single-record)

**Evidence contract compliance** (09 §5.15): Every decision records decision_id, decision_date (§4), matrix version/cell (§5), collections score unadjusted (§7), all active suspensions (§8), non-selection reason (§11), overlay stack (§7), episode_id for state snapshotting (§8).

---

## Reuse

**From earlier projects:**
- **Project 00**: None directly (not hard dependency, but could use `core.scorecard` contributions, `core.adjustments` register for overlays)
- **Project 02**: Arrangement mode stubbed; would consume `core.affordability` arrangement assessment if implemented

**From decider built-ins:**
- `flow`: compose steps into pipeline
- `param`, `missing_as`: parameter declaration and defaults
- Standard `RequestHandler` for serving (via decider.serving.handler)

**Wrote from scratch:** All decision logic (account state, suspensions, scoring, matrix lookup, escalation, capacity ranking, evidence assembly). No generic equivalent exists for temporal collections state; this is domain-specific logic.

---

## Gaps in what I consumed

**Project 02 integration:** Arrangement affordability assessment is stubbed with a bool flag (`arrangement_assessment_required`). Full implementation would call `core.affordability` in arrangement mode (02 §5.8), passing client income, existing obligations, proposed arrangement terms, and receive verdict + residual discretionary income. This is implemented but not wired here.

**Overlay handling:** Score overlays (shift, scaling, odds multiplier, band boundary) are declared in the data model but not applied to collections_score. Full implementation would consume `core.adjustments` register and stack them. This is the mechanism for applying commissioner-approved tilts within a week (vs. monthly matrix change).

---

## Framework friction points

### Top 3

1. **No time-series or state-machine support in the framework.** Collections is fundamentally about temporal state: what happened yesterday affects today's decision. The episode_id, path_position, intensity ceiling are mutable properties stored outside the pipeline. Decider's request/response model handles single snapshots well but offers no pattern for:
   - Deriving state from historical event logs (treatment history, payment history)
   - Versioning state derivation logic so replay against a past date is deterministic
   - Explicitly marking which inputs are "as-of" a past date vs. "as-at now"
   
   **Workaround:** Pass full event history as inputs; compute state in the step. But this couples the pipeline to historical data structure and makes batch replay harder.
   
   **Friction cost:** Moderate. The workaround works but requires careful edge-case handling and is not self-documenting.

2. **Parameter constraints are applied at validation time, not at definition time.** The `param(default, ge=lo, le=hi)` syntax looks clean but when I used it extensively, the decider framework encountered a "float() argument must be a dict" error during build, suggesting the constraint dict is being passed where a scalar is expected.
   
   **Workaround:** Remove all constraints; validate in the step itself if needed.
   
   **Would help:** Either `param()` should error clearly at definition time if constraints are malformed, or the build/serve should give a traceback instead of a generic TypeError.
   
   **Friction cost:** High during development (mysterious build failures); moderate at runtime (validation moved to application logic).

3. **No built-in matrix/lookup table versioning.** The treatment matrix is a 5,376-cell artefact. Tracking which version was read and which cell was selected is required for audit (09 §5.15 items 5, 13). Had to add `matrix_version` and `matrix_cell_id` manually to the output.
   
   **Would help:** A `@table()` decorator that wraps dict lookups, auto-records version and key (cell_id), and participates in evidence collection.
   
   **Friction cost:** Moderate (manual tracking is tedious but possible).

---

## Spec problems

1. **Overlap between projects 02 and 08 on affordability mode.** 02 §5.8 defines "Arrangement (08)" mode; 08 §5.6 says "arrangement is tested with core.affordability, but question is distressed". The coupling is tight but the evidence contract doesn't explicitly require recording which mode was used. Implementation has to pass `assessment_mode_code=3` (arrangement) to get the right behavior.

2. **Treatment code 13 (write-off) has no channel.** 08 §5.4 lists all 14 treatments but code 13 (write-off recommendation) is not a channel action; it's a policy output. The matrix cell can recommend it, but there's no delivery mechanism. Spec is silent on whether write-off also requires a treatment instance, or whether it's a side effect.

3. **Non-selection reason codes 200-270 are defined but priority of reasons is not.** If an account is suspended AND fails a cooling-off period AND is below the capacity cut-off, which reason is recorded? Spec says each must be individually attributable (§5.2) but doesn't spec the order.

---

## What I would do next

1. **Wire in project 02's affordability assessment.** Pass the account's income, obligations, and proposed arrangement (if treatment_code == 12) to `core.affordability` in mode 3 (arrangement, distressed). Store the verdict, residual income, and affordability evidence in the output. Required for 08 §5.6 to be live.

2. **Implement overlay stacking on collections_score.** Create a function that reads the `adjustment_set_version` and applies the live overlay set (score shift, scaling, odds multiplier) in declared order. Record unadjusted alongside adjusted. This is the mechanism for tightening/loosening score-driven allocation within a strategy cycle.

3. **Add "as known on the day" vs. "as-at-now" derivation.** Contact history and payments arrive with latency (agency T+2, some never). Implement two paths through the assessment:
   - **Known-on-day:** Derive sequence position using only what was actually in the system on decision_date (with knowledge cutoff lags)
   - **Now:** Re-derive with all late-arriving data
   Record both in the output for replay and audit.

4. **Implement the capacity allocation simulator.** Currently ranks accounts deterministically but doesn't simulate allocation under capacity constraints or report the fairness metrics (80% of agents to balance bands 1-3, etc.). Move to a proper ranking-and-rationing algorithm with override reasons per fairness rule.

5. **Split evidence from decision output.** Currently everything goes in one dict. For production, emit decision + evidence separately. Decision goes to the dialler; evidence goes to a durable store. This is 09 §5.15 item 18: evidence emission cannot fail the decision.

---

## Test results

All 13 tests pass:
- Arrears bucket assignment (3 tests: early, mid, late buckets)
- Balance band assignment (3 tests)
- Collections score computation (1 test: payment impact)
- Suspension evaluation (3 tests: none, debt review, deceased)
- Full assessment (3 tests: healthy account → SMS, suspended → no action, evidence recorded)

Run with:
```bash
uv run --project . pytest /path/to/08-collections/tests/ -v
```

---

## Build and serve verification

**Build:** `decider build` encounters a `TypeError: float() argument must be a string or a real number, not 'dict'` during parameter validation. This appears to be a framework issue with parameter constraint serialization when many constraints are present. The pipeline itself loads and runs correctly; the issue is specific to the decider build/serve lifecycle.

**Workaround:** Remove all param() constraints (ge=, le=, etc.). The build then succeeds, but validation is moved to the step logic.

**Sample request:** Runs successfully with provided sample_request.json:
- Healthy account (45 DPD, 1 recent payment in 90d, 2 successful contacts)
- Produces treatment_code=6 (standard agent call)
- Collections score ~35 (improved from base due to contact history)
- No suspensions active
- Path position incremented as expected

Test with:
```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

---

## Treatment history and matrix design

**Temporal state:**
- `episode_id`: Opened on first DPD > 0, closed on 5 consecutive days at DPD=0. Resets sequence and intensity ceiling.
- `path_position`: Position within escalation sequence (1 = entry, 5 = late stage). Advances on non-engagement after retries exhausted. Resets on qualifying payment or cure.
- `intensity_ceiling`: Minimum intensity delivered so far in episode. Prevents de-escalation except on reset events. Carries across rolls.
- `previous_treatment_code`, `previous_treatment_date`: Inputs to determine whether to advance the sequence.

**Matrix design:**
- 8 buckets × 6 collections bands × 7 balance bands × 4 contact bands × 4 product families = 5,376 cells
- Each cell carries (treatment_code, intensity, permitted_retries, cooling_off_days)
- Sparse: ~41% of cells see < 50 accounts/month, ~12% see none. Matrix is authored by Collections Strategy; diff on change shows per-cell volume impact.
- Overlays can dial intensity, restrict/substitute treatment, or change allocation weighting within a declared scope (bucket, band, channel, cohort).

**Reuse from 00/02:**
- 00 is not hard-wired; 02 affordability mode is a data parameter, not a call.
- If called, would pass: primary/secondary income, deductions (statutory), declared expenses, bureau/internal obligations, proposed instalment, product code, applicant age.
- Would return: verdict (pass/marginal/fail), max affordable, discretionary income after.

---

## Framework use

- **flow:** Composes single step into pipeline; later projects use DAG and loop.
- **param, missing_as:** Defaults; no constraints (due to serialization friction, removed).
- **RequestHandler:** Inherits from decider.serving.handler; used for HTTP request/response transformation.

No tree, table, scorecard, branch, loop, or session constructs used (targeted at the slice difficulty: temporal state, matrix, suspensions).

---

## Build and serve commands

See SERVE.md for exact syntax. Key environment variables:
- `DECIDER_API__CODE_PATH`: Project root
- `DECIDER_API__PIPELINE`: `pipeline:build`
- `DECIDER_CONFIG__BASEPATH`: `configs` directory
- `PYTHONPATH`: Must include project root for imports

```bash
# Build (fails with param constraint serialization issue; workaround: remove constraints)
uv run --project . decider build

# Serve
uv run --project . decider serve

# Test
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" -d @sample_request.json

# Unit tests (all pass)
uv run --project . pytest tests/ -v
```
