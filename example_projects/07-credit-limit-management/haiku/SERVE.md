# Serving credit limit management

Project 07 implements stages 5.1–5.8 of the monthly limit management programme:
population snapshot, hard exclusions, behaviour scoring, matrix lookup with 
cycle-dial overlays, cap application, affordability reassessment (degraded-evidence 
mode, consuming 02), and portfolio budget allocation. Simulation uses the same 
implementation as production (§5.10, AC3).

## Build

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:/path/to/scratch/haiku/02-affordability:/path/to/scratch/haiku/07-credit-limit-management:$PYTHONPATH
export DECIDER_API__CODE_PATH=/path/to/scratch/haiku/07-credit-limit-management
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/path/to/scratch/haiku/07-credit-limit-management/configs

uv run --project . decider build
```

## Serve

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:/path/to/scratch/haiku/02-affordability:/path/to/scratch/haiku/07-credit-limit-management:$PYTHONPATH
export DECIDER_API__CODE_PATH=/path/to/scratch/haiku/07-credit-limit-management
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/path/to/scratch/haiku/07-credit-limit-management/configs

uv run --project . decider serve
```

## Test with sample request

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Run pytest

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:/path/to/scratch/haiku/02-affordability:/path/to/scratch/haiku/07-credit-limit-management:$PYTHONPATH

uv run --project . pytest /path/to/scratch/haiku/07-credit-limit-management/tests/ -v
```

## Configuration version

Current config version: `0.0.0`
Configuration file: `configs/0.0.0/params.json`

To create a new version, copy the configs directory and increment the version:
```bash
mkdir -p configs/0.0.1
cp configs/0.0.0/params.json configs/0.0.1/params.json
# Edit configs/0.0.1/params.json as needed
```

## Pipeline stages

1. **population_snapshot** (§5.1) — Account state: balance, utilisation, arrears, spend
2. **hard_exclusions** (§5.2) — Evaluate 16 exclusion rules (simplified to core 8)
3. **behaviour_score** (§5.3) — Behaviour scorecard, grading 1–12, PD calibration
4. **matrix_lookup** (§5.4) — 4-dimension assignment matrix, cycle-dial overlay
5. **caps** (§5.5) — Apply cap waterfall (C1 product max, C2 income-multiple, C5 spend)
6. **affordability** (§5.6) — Degraded-evidence mode; reuses project 02 logic
7. **ranking** (§5.8) — Risk-adjusted return ranking for allocation
8. **allocation** (§5.8) — Portfolio budget allocation (simplified single-account stub)

## Key implementation notes

- **Matrix**: 12 grades × 8 utilisation × 6 mob × 2 products = 1 152 cells (generated).
- **Cycle dial**: Multiplier applied to matrix excess over 1.00 (80% dial on 1.50 → 1.40).
- **Overlays**: Stub with cycle_dial_multiplier parameter (full overlay stack in 07-C AC6).
- **Affordability**: Calls project 02 logic in limit-increase mode (mode 2); different buffer.
- **Allocation**: Simplified ranking + budget exhaustion (full fairness floors in 07-C AC5).
- **Simulation**: Same implementation as production; no separate simulation code.

## Evidence contract (09 §5.15, 07-C item 1)

Per-account decision emits:
- `decision_id`, `decision_date` (timestamped)
- `matrix_cell_id`, cell values (authored + adjusted)
- All cap computed values, binding cap code (required for §9 audit item 3)
- `score_unadjusted`, `probability_of_default_unadjusted` (overlay stack visible)
- `affordability_verdict_code`, evidence tier, income staleness
- `allocation_rank`, ranking value, outcome code, reason codes
- Overlay stack in force (07-C AC4, AC6)

## Test results

All tests in `tests/test_pipeline.py` pass:
- Account exclusion rules (no consent, too young, at product max)
- Behaviour scoring with grade migration based on account characteristics
- Matrix lookup with utilisation and mob banding
- Cycle dial overlay application (reduces excess over 1.00)
- Cap application: product max, income-multiple, spend cap
- Affordability verdicts: pass, fail, marginal
- Ranking values (risk-adjusted return calculation)
- Allocation outcomes (funded vs below line)
- Full pipeline composition

Run with:
```bash
uv run --project . pytest /path/to/07-credit-limit-management/tests/ -v
```

## Build and serve verification

✓ `decider build` succeeds: schema validation, 8-step pipeline, params loaded
✓ Sample request scores through handler: response with all 8 step outputs
✓ Matrix generates 1 152 cells at runtime (cached lookup per-product-grade-util-mob)
✓ Reproducibility: same inputs + params + decision_date → identical outputs

See NOTES.md for framework friction, gaps in consumed projects, and next steps.
